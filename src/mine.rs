use std::{sync::Arc, time::Instant};

use colored::*;
use core_affinity::CoreId;
use drillx::{
    equix::{self},
    Hash, Solution,
};
use ore_api::{
    consts::{BUS_ADDRESSES, BUS_COUNT, EPOCH_DURATION},
    state::{Config, Proof},
};
use rand::Rng;
use solana_program::pubkey::Pubkey;
use solana_rpc_client::spinner;
use solana_sdk::signer::Signer;

use crate::{
    args::MineArgs,
    send_and_confirm::ComputeBudget,
    utils::{amount_u64_to_string, get_clock, get_config, get_proof_with_authority, proof_pubkey},
    Miner,
};


const INDEX_SPACE: usize = 65536;

fn hashspace_size() -> usize {
    unsafe { drillx_gpu::BATCH_SIZE as usize * INDEX_SPACE }
}

impl Miner {
    pub async fn mine(&self, args: MineArgs) {
        // Register, if needed.
        let signer = self.signer();
        self.open().await;

        // Check num threads
        self.check_num_cores(args.threads);

        // Start mining loop
        loop {
            // Fetch proof
            let config = get_config(&self.rpc_client).await;
            let proof = get_proof_with_authority(&self.rpc_client, signer.pubkey()).await;
            println!(
                "\nStake: {} ORE\n  Multiplier: {:12}x",
                amount_u64_to_string(proof.balance),
                calculate_multiplier(proof.balance, config.top_balance)
            );

            // Calc cutoff time
            let cutoff_time = self.get_cutoff(proof, args.buffer_time).await;

            println!("Mining for {} seconds", cutoff_time);

            // Run drillx
            let config = get_config(&self.rpc_client).await;

            let mut min_difficulty = args.min_difficulty as u32;
            if min_difficulty == 0 {
                min_difficulty = config.min_difficulty as u32;
            }

            println!("Min difficulty: {}", min_difficulty);

            let solution = Self::find_hash_par(
                proof,
                cutoff_time,
                args.threads,
                min_difficulty,
            )
            .await;

            // Submit most difficult hash
            let mut compute_budget = 500_000;
            let mut ixs = vec![ore_api::instruction::auth(proof_pubkey(signer.pubkey()))];
            if self.should_reset(config).await && rand::thread_rng().gen_range(0..100).eq(&0) {
                compute_budget += 100_000;
                ixs.push(ore_api::instruction::reset(signer.pubkey()));
            }
            ixs.push(ore_api::instruction::mine(
                signer.pubkey(),
                signer.pubkey(),
                find_bus(),
                solution,
            ));
            self.send_and_confirm(&ixs, ComputeBudget::Fixed(compute_budget), false)
                .await
                .ok();
        }
    }


    async fn find_hash_par(
        proof: Proof,
        cutoff_time: u64,
        threads: u64,
        min_difficulty: u32,
    ) -> Solution {
        // Dispatch job to each thread
        let progress_bar = Arc::new(spinner::new_progress_bar());
        progress_bar.set_message("Mining...");


        // test gpu
        let nonce = [0; 8];
        let mut hashes = vec![0u64; hashspace_size()];
        
        unsafe {
            drillx_gpu::hash(proof.challenge.as_ptr(), nonce.as_ptr(), hashes.as_mut_ptr() as *mut u64);
        }

        let challenge = Arc::new(proof.challenge);
        let hashes = Arc::new(hashes);
        let nonce = u64::from_le_bytes(nonce);

        println!("nonce: {}", nonce);

        let chunk_size = unsafe {
            drillx_gpu::BATCH_SIZE as usize / threads as usize   
        };

        println!("chunk_size: {}", chunk_size);

        let handles: Vec<_> = (0..threads)
            .map(|i| {
                let challenge = challenge.clone();
                let hashes = hashes.clone();

                std::thread::spawn({
                    let core_id = i as usize;
                    let core_id = CoreId { id: core_id};
                    println!("binding to cpu: {:?}", core_id);
                    let binding_success = core_affinity::set_for_current(core_id);
                    if !binding_success {
                        println!("Failed to bind to CPU {}", i);
                    }


                    move || {
                        let mut best_nonce = 0;
                        let mut best_difficulty = 0;
                        let mut best_hash = Hash::default();

                        let start = i * chunk_size as u64;
                        let end = if i == threads - 1 {
                            unsafe{ drillx_gpu::BATCH_SIZE as u64}
                        } else {
                            start + chunk_size as u64
                        };


                        for idx in start..end {
                        let mut digest = [0u8; 16];
                        let mut sols = [0u8; 4];
                        let solution = unsafe {
                            let batch_start = hashes.as_ptr().add((i * INDEX_SPACE as u64) as usize);
                            drillx_gpu::solve_all_stages(batch_start,  digest.as_mut_ptr(), sols.as_mut_ptr() as *mut u32);
                            if u32::from_le_bytes(sols).gt(&0) {
            
                                Some(Solution::new(digest, (nonce + idx as u64).to_le_bytes()))

                            } else {
                                None
                            }
                        };

                        if let Some(solution) = solution {
                            let is_valid = solution.is_valid(&challenge);
                            if !is_valid {
                                println!("Invalid solution");
                                continue;
                            }

                            let hash = solution.to_hash();
                            let difficulty = hash.difficulty();
                            if difficulty > best_difficulty {
                                best_difficulty = difficulty;
                                best_nonce = idx as u64;
                                best_hash = hash;

                                println!("Best hash: {} (difficulty: {})", bs58::encode(best_hash.h).into_string(), best_difficulty);
                            }
                        }
                    }

                        (best_nonce, best_difficulty, best_hash)
                    }                   

                })
            })
            .collect();

        // Join handles and return best nonce
        let mut best_nonce = 0u64;
        let mut best_difficulty = 0;
        let mut best_hash = Hash::default();
        for h in handles {
            if let Ok((nonce, difficulty, hash)) = h.join() {
                if difficulty > best_difficulty {
                    best_difficulty = difficulty;
                    best_nonce = nonce;
                    best_hash = hash;
                }
            }
        }

        // Update log
        progress_bar.finish_with_message(format!(
            "Best hash: {} (difficulty: {})",
            bs58::encode(best_hash.h).into_string(),
            best_difficulty
        ));

        Solution::new(best_hash.d, best_nonce.to_le_bytes())
    }

    pub fn check_num_cores(&self, threads: u64) {
        // Check num threads
        let num_cores = num_cpus::get() as u64;
        if threads.gt(&num_cores) {
            println!(
                "{} Number of threads ({}) exceeds available cores ({})",
                "WARNING".bold().yellow(),
                threads,
                num_cores
            );
        }
    }

    async fn should_reset(&self, config: Config) -> bool {
        let clock = get_clock(&self.rpc_client).await;
        config
            .last_reset_at
            .saturating_add(EPOCH_DURATION)
            .saturating_sub(5) // Buffer
            .le(&clock.unix_timestamp)
    }

    async fn get_cutoff(&self, proof: Proof, buffer_time: i64) -> u64 {
        let clock = get_clock(&self.rpc_client).await;
        proof
            .last_hash_at
            .saturating_add(60)
            .saturating_sub(buffer_time)
            .saturating_sub(clock.unix_timestamp)
            .max(0) as u64
    }
}

fn calculate_multiplier(balance: u64, top_balance: u64) -> f64 {
    1.0 + (balance as f64 / top_balance as f64).min(1.0f64)
}

// TODO Pick a better strategy (avoid draining bus)
fn find_bus() -> Pubkey {
    let i = rand::thread_rng().gen_range(0..BUS_COUNT);
    BUS_ADDRESSES[i]
}
