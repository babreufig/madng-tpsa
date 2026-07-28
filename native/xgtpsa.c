#include "xgtpsa.h"

#include "mad_desc_impl.h"
#include "mad_tpsa_impl.h"

int xgtpsa_check_tpsa_compatibility(const tpsa_t *a, const tpsa_t *b)
{
  if (!a || !b) return 0;
  return IS_COMPAT(a, b);
}
