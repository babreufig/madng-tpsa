#ifndef MADNG_LOG_H
#define MADNG_LOG_H

#ifdef __cplusplus
extern "C" {
#endif

/* Install a thread-local callback for fatal libgtpsa errors. */
typedef void (*madng_tpsa_error_handler)(const char *location, const char *message);
madng_tpsa_error_handler madng_tpsa_set_error_handler(madng_tpsa_error_handler handler);

#ifdef __cplusplus
}
#endif

#endif // MADNG_LOG_H
