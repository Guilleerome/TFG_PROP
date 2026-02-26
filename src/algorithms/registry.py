from src.constructors import constructor as construct
from src.improvers import local_search as ls


CONSTRUCTORS = {
    'random': construct.constructor_random,
    'greedy': construct.constructor_greedy,
    'guillermo': construct.constructor_guillermo,
    'greedy_random_by_row': construct.constructor_greedy_random_by_row,
    'greedy_random_global': construct.constructor_greedy_random_global,
    'greedy_random_row_balanced': construct.constructor_greedy_random_row_balanced,
    'random_greedy_by_row': construct.constructor_random_greedy_by_row,
    'random_greedy_global': construct.constructor_random_greedy_global,
    'random_greedy_row_balanced': construct.constructor_random_greedy_row_balanced,
    'global_score_ordering': construct.constructor_global_score_ordering,
    'global_score_ordering_random': construct.constructor_global_score_ordering_random,
}

LOCAL_SEARCHES = {
    'first_move': ls.first_move,
    'best_move': ls.best_move,
    'first_move_swap': ls.first_move_swap,
    'best_move_swap': ls.best_move_swap,
    'none': None,  # Sin búsqueda local
}
