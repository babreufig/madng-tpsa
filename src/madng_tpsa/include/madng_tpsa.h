#ifndef MADNG_TPSA_H
#define MADNG_TPSA_H

#ifdef __cplusplus
extern "C" {
#endif

#include "mad_tpsa.h"

/* Helper functions for this library */
int madng_tpsa_check_tpsa_compatibility(const tpsa_t *left, const tpsa_t *right);
int madng_tpsa_tpsa_variable_index(const tpsa_t *series);
int madng_tpsa_tpsa_single_monomial(
    const tpsa_t *series,
    int monomial_length,
    ord_t monomial_orders[]
);

#ifdef __cplusplus
}
#endif

#endif // MADNG_TPSA_H
