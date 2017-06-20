//
// Created by aomellinger on 6/20/17.
//

#ifndef SRC_DEBUG_H
#define SRC_DEBUG_H

#if GRAPHBLAS_DEBUG

#define GRB_LOG_E(x) do { std::cerr << "ERROR: " << x << std::endl; } while(0)
#define GRB_LOG_I(x) do { std::cerr << "INFO:  " << x << std::endl; } while(0)
#define GRB_LOG_FORMAT(fmt, ...) do { printf(fmt, ##__VA_ARGS__) } while(0)

#else

#define GRB_LOG_E(x)
#define GRB_LOG_I(x)
#define GRB_LOG_FORMAT(fmt, ...)

#endif

#endif //SRC_DEBUG_H
