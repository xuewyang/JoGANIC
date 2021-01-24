from TGNC.utils.functional import softmax
from TGNC.utils.logger import setup_logger
from TGNC.utils.options import eval_str_list
from TGNC.utils.state import get_incremental_state, set_incremental_state
from TGNC.utils.tensor import fill_with_neg_inf, strip_pad
