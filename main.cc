#include "dbscan_nonsparse.h"
#include "dbscan_sparse.h"
#include <fstream>
#include <tuple>
#include <map>
#include <memory>

enum ExitValues {
    Success,
    Help,
    BadArguments,
    IOError
};

typedef std::tuple<std::string, std::string> argtuple_t;

template <typename TNum>
std::vector<TNum> read_corpus(const std::string& input_path, 
        libdbscan::index_t& rows, libdbscan::index_t& cols) {
    std::ifstream s;
    s.exceptions(s.failbit);
    s.open(input_path);
    std::string line;
    std::vector<TNum> values;
    rows = 0;
    cols = 0;
    bool read_cols = false;

    while (std::getline(s, line)) {
        std::string val;
        TNum v;
        std::istringstream ss_line(line);
        while (std::getline(ss_line, val, ',')) {
            if (!read_cols) {
                ++cols;
            }
            values.push_back(std::stof(val));
        }
        read_cols = true;
        ++rows;
    }
    return values;
}

template <typename TNum>
std::unique_ptr<libdbscan::dbscan<TNum> > create_dbscan(const std::string& array_type, 
        const std::string& distance_metric, const std::vector<TNum>& corpus,
        libdbscan::index_t rows, libdbscan::index_t cols) {
    // Steal the buffer from the corpus std::vector - this means it must
    // outlive the dbscan object we are returning. Might have to make a copy of
    // it if this becomes unwieldy, but for now, it neatly avoids a copy
    const TNum* corpus_buf = &corpus[0];

    std::map<argtuple_t,
             std::function<std::unique_ptr<libdbscan::dbscan<TNum>>()>> map {
        { 
            argtuple_t("nonsparse", "euclidean"), 
            [&] () {
                return std::make_unique<libdbscan::dbscan_nonsparse<TNum>>(
                        corpus_buf, rows, cols);
            }
        },
        {
            argtuple_t("sparse", "euclidean"),
            [&] () {
                return std::make_unique<libdbscan::dbscan_sparse<TNum>>(
                        corpus_buf, rows, cols);
            }
        },
        {
            argtuple_t("sparse", "cosine"),
            [&] () {
                return std::make_unique<libdbscan::dbscan_sparse<TNum,
                    libdbscan::cosine_similarity_metric<TNum>>>(corpus_buf, rows, cols);
            }
        }
    };

    auto iter = map.find(argtuple_t(array_type, distance_metric));
    if (iter == map.end()) {
        std::ostringstream o;
        o << "Unknown arguments " << array_type << ", " 
            << distance_metric << std::endl;
        throw std::invalid_argument(o.str());
    }
    return iter->second();
}

template <typename TNum>
int run_dbscan(double eps,
        libdbscan::index_t min_pts,
        const std::string& array_type, 
        const std::string& distance_metric, 
        const std::string& input_path) {

    try {
        libdbscan::index_t rows, cols;
        auto corpus = read_corpus<TNum>(input_path, rows, cols);
        auto dbscan = create_dbscan<TNum>(array_type, distance_metric, 
            corpus, rows, cols);
        std::vector<libdbscan::index_t> results, noise;
        dbscan->run(eps, min_pts, results, noise);
        for (auto& cluster_id : results) {
            std::cout << cluster_id << std::endl;
        }
        return ExitValues::Success;
    } catch (const std::invalid_argument& e) {
        std::cerr << e.what() << std::endl;
        return ExitValues::BadArguments;
    } catch (std::ifstream::failure e) {
        std::cerr << "Error reading " << input_path << " " << e.what() << std::endl;
        return ExitValues::IOError;
    }
}

int main(int argc, char** argv) {
    // this would be more easily done with boost::program_options, but want to
    // avoid boost dependency if possible especially given
    // boost::program_options isn't header-only

    // the arguments we are attempting to populate from argv
    double eps;
    libdbscan::index_t min_pts;
    std::string array_type;
    std::string distance_metric;
    std::string input_path;
    std::string precision;

    const char* usage = 
        "usage: dbscan eps min_pts array_type distance_metric "
        "precision input_path\n"
        "  eps:        parameter to dbscan algorithm, e.g. 0.3\n"
        "  min_pts:    parameter to dbscan algorithm, e.g. 10\n"
        "  array_type: can be sparse or nonsparse\n"
        "  distance_metric: can be euclidean or cosine\n"
        "  precision:  can be double or single\n"
        "  input_path: is the path of a CSV containing vectors";

    if (argc < 7) {
        std::cerr << usage << std::endl;
        return ExitValues::Help;
    }

    eps = std::atof(argv[1]);
    if (eps <= 0.0) {
        std::cerr << "eps must be > 0" << std::endl;
        return ExitValues::BadArguments;
    }

    min_pts = std::atoi(argv[2]);
    if (min_pts <= 0) {
        std::cerr << "min_pts must be > 0" << std::endl;
        return ExitValues::BadArguments;
    }

    array_type = argv[3];
    distance_metric = argv[4];
    precision = argv[5];
    input_path = argv[6];

    if (precision == "double") {
        return run_dbscan<double>(eps, min_pts, array_type, 
                distance_metric, input_path);
    } else {
        return run_dbscan<float>(eps, min_pts, array_type, 
                distance_metric, input_path);
    }
}
