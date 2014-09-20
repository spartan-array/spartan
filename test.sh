#/bin/bash

set -x

test_case=$1
profile_worker=0
profile_master=0
dump_timer=0
worker_list=$3
assign_mode=BY_CORE
log_level=WARN
cluster=1
optimization=1
opt_parakeet_gen=0
opt_auto_tiling=$4
opt_rotate_slice=0
opt_map_fusion=1
load_balance=0
#hosts=beaker-20:8,beaker-21:8,beaker-22:8,beaker-23:8,beaker-24:8,beaker-25:8,beaker-17:4,beaker-19:4,beaker-14:4,beaker-15:4,beaker-16:4,beaker-18:4
#hosts=beaker-23:8,beaker-21:8,beaker-22:8,beaker-25:8
hosts=ip-172-31-10-89.ec2.internal:2,ip-172-31-6-155.ec2.internal:2,ip-172-31-3-31.ec2.internal:2,ip-172-31-3-109.ec2.internal:2,ip-172-31-15-39.ec2.internal:2,ip-172-31-10-115.ec2.internal:2,ip-172-31-1-247.ec2.internal:2,ip-172-31-13-146.ec2.internal:2,ip-172-31-8-34.ec2.internal:2,ip-172-31-11-217.ec2.internal:2,ip-172-31-4-198.ec2.internal:2,ip-172-31-10-135.ec2.internal:2,ip-172-31-2-139.ec2.internal:2,ip-172-31-1-36.ec2.internal:2,ip-172-31-8-183.ec2.internal:2,ip-172-31-13-13.ec2.internal:2,ip-172-31-12-21.ec2.internal:2,ip-172-31-12-231.ec2.internal:2,ip-172-31-5-136.ec2.internal:2,ip-172-31-4-229.ec2.internal:2,ip-172-31-14-214.ec2.internal:2,ip-172-31-5-153.ec2.internal:2,ip-172-31-13-11.ec2.internal:2,ip-172-31-12-96.ec2.internal:2,ip-172-31-5-155.ec2.internal:2,ip-172-31-9-36.ec2.internal:2,ip-172-31-1-149.ec2.internal:2,ip-172-31-4-56.ec2.internal:2,ip-172-31-8-68.ec2.internal:2,ip-172-31-8-99.ec2.internal:2,ip-172-31-4-214.ec2.internal:2,ip-172-31-13-36.ec2.internal:2,ip-172-31-14-43.ec2.internal:2,ip-172-31-5-165.ec2.internal:2,ip-172-31-0-34.ec2.internal:2,ip-172-31-0-77.ec2.internal:2,ip-172-31-14-97.ec2.internal:2,ip-172-31-6-124.ec2.internal:2,ip-172-31-1-221.ec2.internal:2,ip-172-31-11-48.ec2.internal:2,ip-172-31-13-63.ec2.internal:2,ip-172-31-3-125.ec2.internal:2,ip-172-31-12-108.ec2.internal:2,ip-172-31-13-74.ec2.internal:2,ip-172-31-13-128.ec2.internal:2,ip-172-31-6-80.ec2.internal:2,ip-172-31-12-120.ec2.internal:2,ip-172-31-5-219.ec2.internal:2,ip-172-31-4-111.ec2.internal:2,ip-172-31-13-48.ec2.internal:2,ip-172-31-11-86.ec2.internal:2,ip-172-31-5-69.ec2.internal:2,ip-172-31-10-254.ec2.internal:2,ip-172-31-6-230.ec2.internal:2,ip-172-31-13-242.ec2.internal:2,ip-172-31-7-71.ec2.internal:2,ip-172-31-13-126.ec2.internal:2,ip-172-31-13-121.ec2.internal:2,ip-172-31-13-166.ec2.internal:2,ip-172-31-10-41.ec2.internal:2,ip-172-31-1-206.ec2.internal:2,ip-172-31-14-67.ec2.internal:2,ip-172-31-13-76.ec2.internal:2,ip-172-31-8-217.ec2.internal:2,ip-172-31-0-135.ec2.internal:2,ip-172-31-11-184.ec2.internal:2,ip-172-31-1-46.ec2.internal:2,ip-172-31-5-201.ec2.internal:2,ip-172-31-12-254.ec2.internal:2,ip-172-31-4-245.ec2.internal:2,ip-172-31-13-177.ec2.internal:2,ip-172-31-13-221.ec2.internal:2,ip-172-31-14-18.ec2.internal:2,ip-172-31-3-139.ec2.internal:2,ip-172-31-2-26.ec2.internal:2,ip-172-31-4-131.ec2.internal:2,ip-172-31-0-100.ec2.internal:2,ip-172-31-7-157.ec2.internal:2,ip-172-31-9-88.ec2.internal:2,ip-172-31-3-170.ec2.internal:2,ip-172-31-7-212.ec2.internal:2,ip-172-31-5-98.ec2.internal:2,ip-172-31-10-57.ec2.internal:2,ip-172-31-7-51.ec2.internal:2,ip-172-31-3-78.ec2.internal:2,ip-172-31-4-237.ec2.internal:2,ip-172-31-6-241.ec2.internal:2,ip-172-31-12-227.ec2.internal:2,ip-172-31-4-105.ec2.internal:2,ip-172-31-3-88.ec2.internal:2,ip-172-31-3-81.ec2.internal:2,ip-172-31-15-227.ec2.internal:2,ip-172-31-5-138.ec2.internal:2,ip-172-31-6-145.ec2.internal:2,ip-172-31-11-65.ec2.internal:2,ip-172-31-8-77.ec2.internal:2,ip-172-31-7-218.ec2.internal:2,ip-172-31-12-31.ec2.internal:2,ip-172-31-8-181.ec2.internal:2,ip-172-31-3-72.ec2.internal:2,ip-172-31-13-8.ec2.internal:2,ip-172-31-5-49.ec2.internal:2,ip-172-31-1-96.ec2.internal:2,ip-172-31-8-205.ec2.internal:2,ip-172-31-4-86.ec2.internal:2,ip-172-31-1-127.ec2.internal:2,ip-172-31-0-23.ec2.internal:2,ip-172-31-10-189.ec2.internal:2,ip-172-31-5-171.ec2.internal:2,ip-172-31-6-136.ec2.internal:2,ip-172-31-9-66.ec2.internal:2,ip-172-31-14-45.ec2.internal:2,ip-172-31-12-25.ec2.internal:2,ip-172-31-3-190.ec2.internal:2,ip-172-31-1-110.ec2.internal:2,ip-172-31-2-181.ec2.internal:2,ip-172-31-10-108.ec2.internal:2,ip-172-31-1-224.ec2.internal:2,ip-172-31-2-220.ec2.internal:2,ip-172-31-11-104.ec2.internal:2,ip-172-31-15-237.ec2.internal:2,ip-172-31-5-97.ec2.internal:2,ip-172-31-7-18.ec2.internal:2,ip-172-31-14-172.ec2.internal:2,ip-172-31-13-41.ec2.internal:2,ip-172-31-0-129.ec2.internal:2,ip-172-31-14-37.ec2.internal:2,ip-172-31-13-145.ec2.internal:2
tile_assignment_strategy=round_robin
default_rpc_timeout=3000000

function kill_all_python_process {
    killall -uchenqi python;
    machines=`echo $hosts | awk '{split($1, s, ",");for(i=1;i<=length(s);i++) {split(s[i], t, ":");print t[1]}}'`
    for machine in $machines
    do
        echo "$machine"
        ssh $machine -f "killall -uchenqi python;ps -uchenqi | grep python;"
    done
}

if [ $2 = "profile" ] 
then
    #kill_all_python_process
    time python $test_case --cluster=$cluster --hosts=$hosts --dump_timer=$dump_timer --use_threads=0 --num_workers=$worker_list --optimization=$optimization --opt_map_fusion=$opt_map_fusion --opt_parakeet_gen=$opt_parakeet_gen --opt_auto_tiling=$opt_auto_tiling --opt_rotate_slice=$opt_rotate_slice --worker_list=$worker_list --assign_mode=$assign_mode --tile_assignment_strategy=$tile_assignment_strategy --default_rpc_timeout=$default_rpc_timeout --profile_worker=$profile_worker --profile_master=$profile_master --log_level=$log_level --load_balance=$load_balance
else
    nosetests -s --nologcapture $test_case
fi
