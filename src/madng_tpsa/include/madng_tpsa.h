#ifndef MADNG_TPSA_H
#define MADNG_TPSA_H

#include "mad_tpsa.h"

int madng_tpsa_check_tpsa_compatibility(const tpsa_t *left, const tpsa_t *right);
int madng_tpsa_tpsa_variable_index(const tpsa_t *series);
int madng_tpsa_tpsa_single_monomial(
    const tpsa_t *series,
    int monomial_length,
    ord_t monomial_orders[]
);

#endif // MADNG_TPSA_H
