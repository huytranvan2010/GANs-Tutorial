## Giới thiệu

Trong phần này chúng ta sẽ xây dựng model GANs trên bộ dữ liệu MNIST. Bộ dữ liệu MNIST nhỏ, đã xử lý cho phép chúng ta train nhanh hơn và có thời gian để tập trung vào việc xây dựng model, hiểu hơn về model. Trong bài này chúng ta sẽ tìm hiểu:
- Cách xây dựng và train độc lập (standalone) discriminator model cho việc học sự khác nhau giữa real và fake images
- Cách xây dựng độc lập generator model và train composite generator và discriminator model (gộp lại với nhau).
- Cách đánh giá performance của mô hình GAN và sử dụng final standalone generator model để sinh ảnh mới

### MNIST Handwritten Digit Dataset

Bộ dữ liệu MNIST có 70000 ảnh có kích thước `28x28` graysclae images các chữ số từ 0 đến 9. 
