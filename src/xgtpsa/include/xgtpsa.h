#ifndef XGTPSA_H
#define XGTPSA_H

#include "mad_tpsa.h"

int xgtpsa_check_tpsa_compatibility(const tpsa_t *left, const tpsa_t *right);
int xgtpsa_tpsa_variable_index(const tpsa_t *series);
int xgtpsa_tpsa_single_monomial(
    const tpsa_t *series,
    int monomial_length,
    ord_t monomial_orders[]
);

#endif // XGTPSA_H
