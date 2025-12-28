#include <iostream>
#include <vector>
#include <string>
#include <fstream> // For file handling
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

// --- Configuration ---
const int MODEL_INPUT_SIZE = 224;
const int MAX_TEXT_LEN = 77;
const std::vector<float> MEAN = {0.48145466f, 0.4578275f, 0.40821073f};
const std::vector<float> STD  = {0.26862954f, 0.26130258f, 0.27577711f};

class ClipEncoder {
private:
    Ort::Env env;
    Ort::Session session;
    Ort::MemoryInfo memory_info;

public:
    ClipEncoder(const std::wstring& model_path) 
        : env(ORT_LOGGING_LEVEL_WARNING, "VideoRAG"), 
          session(nullptr), 
          memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
        
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4); 
        session = Ort::Session(env, model_path.c_str(), session_options);
    }

    std::vector<float> get_image_embedding(const cv::Mat& img) {
        cv::Mat resized_img;
        cv::resize(img, resized_img, cv::Size(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE));
        cv::cvtColor(resized_img, resized_img, cv::COLOR_BGR2RGB);

        std::vector<float> input_tensor_values(1 * 3 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE);
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < MODEL_INPUT_SIZE; h++) {
                for (int w = 0; w < MODEL_INPUT_SIZE; w++) {
                    float pixel = resized_img.at<cv::Vec3b>(h, w)[c] / 255.0f;
                    input_tensor_values[c * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE + h * MODEL_INPUT_SIZE + w] = (pixel - MEAN[c]) / STD[c];
                }
            }
        }

        std::vector<int64_t> input_shape = {1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE};
        std::vector<int64_t> text_input_ids(1 * MAX_TEXT_LEN, 0);
        std::vector<int64_t> text_attention_mask(1 * MAX_TEXT_LEN, 0);
        std::vector<int64_t> text_shape = {1, MAX_TEXT_LEN};

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, text_input_ids.data(), text_input_ids.size(), text_shape.data(), text_shape.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, text_attention_mask.data(), text_attention_mask.size(), text_shape.data(), text_shape.size()));

        const char* input_names[] = {"pixel_values", "input_ids", "attention_mask"};
        const char* output_names[] = {"image_embeds"};

        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, input_tensors.data(), 3, output_names, 1);
        float* floatarr = output_tensors[0].GetTensorMutableData<float>();
        return std::vector<float>(floatarr, floatarr + 512); 
    }
};

void process_and_save(const std::string& video_path, ClipEncoder& encoder, const std::string& output_db) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "❌ Error: Could not open video." << std::endl;
        return;
    }

    // Open file for binary writing
    std::ofstream outfile(output_db, std::ios::binary);
    if (!outfile.is_open()) {
        std::cerr << "❌ Error: Could not create database file." << std::endl;
        return;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    int frame_interval = (int)fps; 
    if (frame_interval == 0) frame_interval = 30;

    cv::Mat frame;
    int frame_count = 0;
    int indexed_count = 0;

    std::cout << "--- Indexing to " << output_db << " ---" << std::endl;

    while (cap.read(frame)) {
        if (frame_count % frame_interval == 0) {
            double timestamp = frame_count / fps;
            
            // 1. Get vector
            std::vector<float> embedding = encoder.get_image_embedding(frame);
            
            // 2. Save to file (Format: [Timestamp: double] + [Vector: 512 floats])
            outfile.write(reinterpret_cast<const char*>(&timestamp), sizeof(double));
            outfile.write(reinterpret_cast<const char*>(embedding.data()), embedding.size() * sizeof(float));
            
            std::cout << "\r[Saved] Time: " << timestamp << "s" << std::flush;
            indexed_count++;
        }
        frame_count++;
    }
    outfile.close();
    std::cout << "\n✅ Database created! Saved " << indexed_count << " entries to " << output_db << std::endl;
}

int main() {
    try {
        ClipEncoder encoder(L"model_quantized.onnx");
        // Database output filename
        process_and_save("test.mp4", encoder, "video_index.bin");
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}