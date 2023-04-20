data_time=$(date "+%Y-%m-%d_%H-%M-%S")
dst_dir=../backup/PROJECT_RK_${data_time}
mkdir -p ${dst_dir}
cp src TESTS tools include config.mk Makefile *.sh  ${dst_dir} -rf

undistort_dir=${dst_dir}/tools/Undistort
# mkdir -p ${undistort_dir}
# cp tools/Undistort/*.py ${undistort_dir}/
# cp tools/Undistort/*.cpp ${undistort_dir}/
# cp tools/Undistort/*.sh ${undistort_dir}/