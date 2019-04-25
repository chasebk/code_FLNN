Ý tưởng của toàn bộ quá trình truy xuất dưx liệu google cluster trace:
  + Lấy ra danh sách Job ID và số lượng task của nó -- để tìm ra giá trị Job id có số lượng task là lơn nhâts két quả: 6336594489
  + Lấy ra toàn bộ giá trị resoure usage của Job ID đó theo từng part.
  + Lấy ra giá trị minstart time và max end time của tưngf part -- chỉ của job id đó.
  + Đọc từng part, Time stamp = minstart , tăng dần 10s sau mỗi vòng lặp đến khi giá trị đó vượt quá max end.
	Xử lí vấn đề chồng lấn thời gian giữa các part, đối với các time stamp nằm trong khoảng chồng lấn, tiến hành truy xuất các gia trị
	resource của nhãn đó tại part trước đó hoặc part sau đó(tùy vào vị trí chồng lấn).
  + Tính tổng resource đối với từng nhãn thời gian.
  + Thêm cột nhãn thời gian và gộp các time stamp file lại để tạo thành timeseries file
  + Tiến hành tính tổng resource phân theo nhãn thời gian-- để xử lí các time stamp chồng lấn và sắp xếp lại file kết quả theo giá trị time stamp tăng dần.



my_sample_resource_usage_TopJobId.csv -- Sau khi xử lí thời gian chồng lấn giữa các part, Gán timeStamp, và gộp file lại. Tại file này, các giá trị thời gian tại các vị trí chồng lấn giữa các part sẽ có thể xuất hiện 2 lần

offical_data_resource_TopJobId.csv -- Chứa resource usage timeseries sau khi đã tiến hành group by các giá trị thời gian nằm trong khoảng chồng lấn giữa 2 part, tuy nhiên chưa được sắp xếp lại thời gian.


my_offical_data_resource_TopJobId.csv  -- Đây là dữ liệu cuôí cùng của resource usage timeseries. File này là kết quả của việc sắp xếp lại thời gian từ file offical_data_resource_TopJobId.csv

==============================================
Các field trong file data:

time_stamp,
numberOftaskIndex,
numberOfmachineId,
meanCPUUsage,
CanonicalMemUsage,
AssignMem,
unmapped_cache_usage,
page_cache_usage,
max_mem_usage,
mean_diskIO_time,
mean_local_disk_space,
max_cpu_usage,
max_disk_io_time,
cpi,
 mai,
sampling_portion,
agg_type,
sampled_cpu_usage
