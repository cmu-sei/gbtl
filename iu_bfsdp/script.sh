#!/usr/bin/env python2
import os
'''
for i in xrange(6,25):
    for j in xrange(0,64):
        #os.system('nvprof ./main '+str(i)+" 16 "+str(j));
        os.system('./main '+str(i)+" 16 "+str(j));

for i in xrange(6,25):
    os.system('./main '+str(i)+" 16 0");
'''

metric_opts=["l1_cache_global_hit_rate","l1_cache_local_hit_rate","sm_efficiency","ipc","achieved_occupancy","gld_requested_throughput","gst_requested_throughput","sm_efficiency_instance","ipc_instance","inst_replay_overhead","shared_replay_overhead","global_replay_overhead","global_cache_replay_overhead","tex_cache_hit_rate","tex_cache_throughput","dram_read_throughput","dram_write_throughput","gst_throughput","gld_throughput","local_replay_overhead","shared_efficiency","gld_efficiency","gst_efficiency","l2_l1_read_hit_rate","l2_texture_read_hit_rate","l2_l1_read_throughput","l2_texture_read_throughput","local_memory_overhead","warp_execution_efficiency","nc_gld_requested_throughput","issued_ipc","inst_per_warp","issue_slot_utilization","local_load_transactions_per_request","local_store_transactions_per_request","shared_load_transactions_per_request","shared_store_transactions_per_request","gld_transactions_per_request","gst_transactions_per_request","local_load_transactions","local_store_transactions","shared_load_transactions","shared_store_transactions","gld_transactions","gst_transactions","sysmem_read_transactions","sysmem_write_transactions","tex_cache_transactions","dram_read_transactions","dram_write_transactions","l2_read_transactions","l2_write_transactions","local_load_throughput","local_store_throughput","shared_load_throughput","shared_store_throughput","l2_read_throughput","l2_write_throughput","sysmem_read_throughput","sysmem_write_throughput","warp_nonpred_execution_efficiency","cf_issued","cf_executed","ldst_issued","ldst_executed","flop_count_sp","flop_count_sp_add","flop_count_sp_mul","flop_count_sp_fma","flop_count_dp","flop_count_dp_add","flop_count_dp_mul","flop_count_dp_fma","flop_count_sp_special","stall_inst_fetch","stall_exec_dependency","stall_memory_dependency","stall_texture","stall_sync","stall_other","l1_shared_utilization","l2_utilization","tex_utilization","dram_utilization","sysmem_utilization","ldst_fu_utilization","alu_fu_utilization","cf_fu_utilization","tex_fu_utilization","inst_executed","inst_issued","issue_slots","nc_l2_read_throughput","nc_l2_read_transactions","nc_cache_global_hit_rate","nc_gld_throughput","nc_gld_efficiency","l2_atomic_throughput","inst_fp_32","inst_fp_64","inst_integer","inst_bit_convert","inst_control","inst_compute_ld_st","inst_misc","inst_inter_thread_communication","atomic_replay_overhead","atomic_transactions","atomic_transactions_per_request","l2_l1_read_transactions","l2_l1_write_transactions","l2_tex_read_transactions","l2_l1_write_throughput","l2_atomic_transactions","ecc_transactions","ecc_throughput","stall_pipe_busy","stall_constant_memory_dependency","flop_sp_efficiency","flop_dp_efficiency","stall_memory_throttle","stall_not_selected","eligible_warps_per_cycle","atomic_throughput"]

for i in xrange(6,25):
  os.system('mkdir -p metrics/'+str(i))
  for opt in metric_opts:
    os.system('nvprof --csv --log-file metrics/'+str(i)+'/metricres_'+opt+'.csv -m '+opt+' ./main '+str(i)+' 16 0 0');
