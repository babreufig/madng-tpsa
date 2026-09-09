#include "madng_tpsa.h"

#include "mad_desc_impl.h"
#include "mad_tpsa_impl.h"

/*
 * Return non-zero when two TPSA objects can be used together in MAD-NG
 * arithmetic operations.
 *
 * MAD-NG calls ensure(...) and terminates the process on incompatible inputs.
 * This helper lets the Python wrapper check compatibility first and raise a
 * normal Python exception instead.
 */
int madng_tpsa_check_tpsa_compatibility(const tpsa_t *left, const tpsa_t *right) {
    if (!left || !right) {
        return 0;
    }

    return IS_COMPAT(left, right);
}

/*
 * Copy the monomial orders from a TPSA containing exactly one unit monomial.
 *
 * Return the underlying coefficient index on success. Return -1 if the series
 * has a constant part, has no non-constant monomial, has more than one
 * non-constant monomial, its coefficient is not one, or the output buffer length
 * does not match the descriptor.
 */
int madng_tpsa_tpsa_single_monomial(
    const tpsa_t *series,
    int monomial_len,
    ord_t monomial_orders[]
) {
    if (!series || !monomial_orders || monomial_len != series->d->nn || series->coef[0] != 0) {
        return -1;
    }

    int coefficient_index = -1;
    TPSA_SCAN(series) {
        if (series->coef[i] == 0) {
            continue;
        }
        if (series->coef[i] != 1 || coefficient_index >= 0) {
            return -1;
        }

        const ord_t *source_orders = series->d->To[i];
        for (int monomial_index = 0; monomial_index < monomial_len; ++monomial_index) {
            monomial_orders[monomial_index] = source_orders[monomial_index];
        }
        coefficient_index = i;
    }

    return coefficient_index;
}
