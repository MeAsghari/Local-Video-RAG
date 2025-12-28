cmake_minimum_required(VERSION 3.17)
project(Local-Video-RAG)

# 1. تنظیم استاندارد C++ (حداقل 17 برای مدرن بودن)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. پیدا کردن OpenCV
# اگر خطا داد، باید مسیر OpenCV_DIR را دستی ست کنی، اما معمولاً خودش پیدا می‌کند
set(OpenCV_DIR "C:/Users/Pc Phonicx/Desktop/Local-Video-RAG/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED)

# 3. تنظیمات ONNX Runtime (بخش حساس)
# ما فرض می‌کنیم پوشه onnxruntime در روت پروژه است
set(ORT_ROOT "${CMAKE_SOURCE_DIR}/onnxruntime")
include_directories(${ORT_ROOT}/include)
link_directories(${ORT_ROOT}/lib)

# 4. تعریف فایل اجرایی
add_executable(Local-Video-RAG main.cpp)

# 5. لینک کردن کتابخانه‌ها به فایل اجرایی
# نکته: اگر لینوکس هستی، شاید اسم فایل libonnxruntime.so باشد (بدون .lib)
if(WIN32)
    target_link_libraries(Local-Video-RAG PRIVATE ${OpenCV_LIBS} onnxruntime)
else()
    target_link_libraries(Local-Video-RAG PRIVATE ${OpenCV_LIBS} onnxruntime)
endif()

# پیام وضعیت برای اینکه بفهمیم همه چیز اوکی است
message(STATUS "OpenCV Found: ${OpenCV_VERSION}")
message(STATUS "ONNX Runtime Root: ${ORT_ROOT}")