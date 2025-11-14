#include <ai_vmm/compute_backend.hpp>
#include <stdexcept>

namespace ai_vmm {

/**
 * @brief Internal model representation
 */
class Model {
public:
    Model() = default;
    virtual ~Model() = default;
    
    /**
     * @brief Get model metadata
     */
    virtual ModelCategory get_category() const = 0;
    virtual std::vector<Precision> get_supported_precisions() const = 0;
    virtual size_t get_memory_requirements() const = 0;
    
    /**
     * @brief Model execution interface
     */
    virtual Tensor execute(const Tensor& input) = 0;
};

/**
 * @brief Model graph representation for compilation
 */
class ModelGraph {
public:
    struct Node {
        std::string op_type;
        std::vector<std::string> inputs;
        std::vector<std::string> outputs;
        std::unordered_map<std::string, std::string> attributes;
    };
    
    ModelGraph() = default;
    
    void add_node(const Node& node) {
        nodes_.push_back(node);
    }
    
    const std::vector<Node>& get_nodes() const {
        return nodes_;
    }
    
    void set_input_shape(const std::string& input_name, const std::vector<size_t>& shape) {
        input_shapes_[input_name] = shape;
    }
    
    void set_output_shape(const std::string& output_name, const std::vector<size_t>& shape) {
        output_shapes_[output_name] = shape;
    }
    
    ModelCategory analyze_category() const {
        // Simple heuristic-based model categorization
        bool has_attention = false;
        bool has_conv = false;
        bool has_rnn = false;
        bool has_embedding = false;
        
        for (const auto& node : nodes_) {
            if (node.op_type == "MultiHeadAttention" || node.op_type == "Attention") {
                has_attention = true;
            } else if (node.op_type == "Conv" || node.op_type == "Conv2D") {
                has_conv = true;
            } else if (node.op_type == "LSTM" || node.op_type == "GRU" || node.op_type == "RNN") {
                has_rnn = true;
            } else if (node.op_type == "Embedding" || node.op_type == "EmbeddingBag") {
                has_embedding = true;
            }
        }
        
        // Categorize based on detected patterns
        if (has_attention && !has_conv) {
            return ModelCategory::LLM_TRANSFORMER;
        } else if (has_attention && has_conv) {
            return ModelCategory::VISION_TRANSFORMER;
        } else if (has_conv) {
            return ModelCategory::VISION_CNN;
        } else if (has_rnn) {
            return ModelCategory::SPEECH_RNN;
        } else if (has_embedding) {
            return ModelCategory::RECOMMENDATION_SYSTEM;
        }
        
        return ModelCategory::UNKNOWN_ARCHITECTURE;
    }
    
private:
    std::vector<Node> nodes_;
    std::unordered_map<std::string, std::vector<size_t>> input_shapes_;
    std::unordered_map<std::string, std::vector<size_t>> output_shapes_;
};

} // namespace ai_vmm