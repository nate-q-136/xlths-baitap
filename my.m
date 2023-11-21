% Tạo dữ liệu mẫu
x = 1:10;  % Dãy số từ 1 đến 10
y1 = 2*x + 5;  % Hàm đường thẳng y = 2x + 5
y2 = x.^2;  % Hàm bậc hai y = x^2

% Biểu đồ đường
figure;
plot(x, y1, 'LineWidth', 2);  % Vẽ đường thẳng y1 với độ dày đường là 2
hold on;  % Giữ lại biểu đồ hiện tại
plot(x, y2, 'r--', 'LineWidth', 2);  % Vẽ đường nét đứt màu đỏ y2 với độ dày đường là 2
xlabel('X-axis');
ylabel('Y-axis');
title('Biểu đồ đường');
legend('y = 2x + 5', 'y = x^2');
grid on;  % Hiển thị lưới

% Biểu đồ điểm
figure;
scatter(x, y1, 100, 'filled');  % Vẽ biểu đồ điểm với kích thước điểm là 100 và điền màu
hold on;
scatter(x, y2, 100, 'r', 'filled');  % Vẽ biểu đồ điểm màu đỏ cho y2 và điền màu
xlabel('X-axis');
ylabel('Y-axis');
title('Biểu đồ điểm');
legend('y = 2x + 5', 'y = x^2');
grid on;
