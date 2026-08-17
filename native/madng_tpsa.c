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
int madng_tpsa_check_tpsa_compatibility(const tpsa_t *left, const tpsa_t *right)
{
  if (!left || !right) {
    return 0;
  }

  return IS_COMPAT(left, right);
}

/*
 * Return the 1-based variable/parameter index represented by an identity TPSA.
 *
 * A valid identity TPSA has zero constant part and exactly one first-order
 * monomial with coefficient 1. Return -1 when the series is not an identity
 * variable.
 */
int madng_tpsa_tpsa_variable_index(const tpsa_t *series)
{
  if (!series || series->coef[0] != 0 || series->lo != 1 || series->hi != 1) {
    return -1;
  }

  int variable_index = 0;
  TPSA_SCAN(series) {
    const idx_t coefficient_index = i;
    const num_t coefficient = series->coef[coefficient_index];
    if (coefficient == 0) {
      continue;
    }
    if (coefficient != 1 || variable_index != 0) {
      return -1;
    }

    const ord_t *monomial_orders = series->d->To[coefficient_index];
    for (int monomial_index = 0; monomial_index < series->d->nn; ++monomial_index) {
      const ord_t variable_order = monomial_orders[monomial_index];
      if (variable_order == 0) {
        continue;
      }
      if (variable_order != 1 || variable_index != 0) {
        return -1;
      }
      variable_index = monomial_index + 1;
    }
  }

  return variable_index > 0 ? variable_index : -1;
}

/*
 * Copy the monomial orders from a TPSA that contains exactly one non-constant
 * monomial.
 *
 * Return non-zero on success. Return 0 if the series has a constant part, has no
 * non-constant monomial, has more than one non-constant monomial, or the output
 * buffer length does not match the descriptor.
 */
int madng_tpsa_tpsa_single_monomial(
    const tpsa_t *series,
    int monomial_len,
    ord_t monomial_orders[]
)
{
  if (!series || !monomial_orders || monomial_len != series->d->nn || series->coef[0] != 0) {
    return 0;
  }

  int found = 0;
  TPSA_SCAN(series) {
    if (series->coef[i] == 0) {
      continue;
    }
    if (found) {
      return 0;
    }

    const ord_t *source_orders = series->d->To[i];
    for (int monomial_index = 0; monomial_index < monomial_len; ++monomial_index) {
      monomial_orders[monomial_index] = source_orders[monomial_index];
    }
    found = 1;
  }

  return found;
}
