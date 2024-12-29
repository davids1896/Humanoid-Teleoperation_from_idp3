# bash convert_data.sh


save_img=1
save_depth=0


demo_path=/home/DavidHong/data/leju/data_new/home/kuavo/rosbag_record/pick-lemon/output_directory_new
save_path=/home/DavidHong/data/leju/converted_data_10hz

cd /home/DavidHong/code/git_clone/Humanoid-Teleoperation/humanoid_teleoperation/scripts
python /home/DavidHong/code/git_clone/Humanoid-Teleoperation/humanoid_teleoperation/scripts/convert_demos_leju.py --demo_dir ${demo_path} \
                                --save_dir ${save_path} \
                                --save_img ${save_img} \
                                --save_depth ${save_depth} \
