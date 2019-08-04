
import math
import numpy as np
from scipy import optimize
from scipy import integrate
import collections

State = collections.namedtuple('State', 'is_active remaining_prob cur_bid cur_value_idx')


class Strategy:
    def __init__(self):
        self.start_points = []
        self.end_points = []
        # keeps track of jump points in CDF
        self.F_jump_points = []  # (bid, F)

    def update(self, start_points, end_points, F_jump_points):
        self.start_points = start_points
        self.end_points = end_points
        # reverse it to make it in ascending order
        self.F_jump_points = F_jump_points[::-1]


class Bidder:
    def __init__(self, values, prob):
        assert len(values) == len(prob), 'The number of bids and the number of probabilities are not equal.'
        assert math.isclose(sum(prob), 1.0), 'Bid probabilities do not add up to 1.'
        sorted_idx = sorted(list(range(len(values))), key=lambda x: values[x])
        sorted_values = [values[i] for i in sorted_idx]
        sorted_prob = [prob[i] for i in sorted_idx]
        idx = [i for i in range(len(values)) if sorted_prob[i] > 0]
        self.values = [sorted_values[i] for i in idx]
        self.prob = [sorted_prob[i] for i in idx]

        self.strategy = Strategy()


def compute_min_winning_bid(bidders):
    """Compute the min winning bid according to Lemma 5.1"""
    min_values = [x.values[0] for x in bidders]
    max_min_value = max(min_values)
    max_min_value_bidder = np.argmax(min_values)

    max_utility = -1.0
    min_winning_bid = -1.0
    for i in range(len(bidders)):
        for j in range(len(bidders[i].values)):
            b = bidders[i].values[j]
            if b > max_min_value:
                break
            winning_prob = 1.0
            for k in range(len(bidders)):
                if k != max_min_value_bidder:
                    winning_prob *= sum(
                        [bidders[k].prob[l] for l in range(len(bidders[k].values)) if bidders[k].values[l] <= b])
            utility = winning_prob * (max_min_value - b)
            if utility > max_utility - 1e-8:
                max_utility = utility
                min_winning_bid = b
    return min_winning_bid


def h(cur_bid, cur_bidder_value, active_values, poly_form=False):
    """h(x) = 1/(sigma - 1)*(sum of 1/(v_j-x)) - 1/(v_i-x).

    Main arguments:
    poly_form: whether to solve the equivalent polynomial or the original form.
    """
    if not poly_form:
        return (sum(1.0 / (np.array(active_values) - cur_bid)) /
                (len(active_values) - 1.0) - 1.0 / (cur_bidder_value - cur_bid))
    else:
        active_values_tiled = np.tile(np.array(active_values), (len(active_values), 1)) - cur_bid
        np.fill_diagonal(active_values_tiled, 1.0)
        return (sum(np.prod(active_values_tiled, axis=1)) * (cur_bidder_value - cur_bid) -
                (len(active_values) - 1.0) * np.prod(np.array(active_values) - cur_bid))


def H(cur_bid, cur_bidder_value, active_values):
    """H(x) = ln(v_i-x) - 1/(sigma - 1)*(sum of ln(v_j-x))"""
    return (np.log(cur_bidder_value - cur_bid) -
            sum(np.log(np.array(active_values) - cur_bid)) / (len(active_values) - 1.0))


def exp_H(cur_bid, cur_bidder_value, active_values):
    """e to the H, used to maintain probability consumptions"""
    return (cur_bidder_value - cur_bid) / (
            np.prod(np.array(active_values) - cur_bid) ** (1.0 / (len(active_values) - 1.0)))


def compute_strategies_given_max_winning_bid(max_winning_bid, bidders):
    """computes bidding strategies given the max winning bid"""
    bid_start_points = [[-1.0] * len(bidder.values) for bidder in bidders]
    bid_end_points = [[-1.0] * len(bidder.values) for bidder in bidders]
    F_jump_points = [[(max_winning_bid, 1.0)] for _ in range(len(bidders))]

    # current state
    is_active = [0] * len(bidders)
    remaining_prob = [-1.0] * len(bidders)
    cur_bid = max_winning_bid
    cur_value_idx = [len(bidder.values) - 1 for bidder in bidders]

    def cur_value(bidder_idx):
        if cur_value_idx[bidder_idx] >= 0:
            return bidders[bidder_idx].values[cur_value_idx[bidder_idx]]
        else:
            return None

    def next_candidate():
        """the next bidder to enter the bidding set"""
        candidate_bidder = -1
        candidate_value = -1
        for n in range(len(bidders)):
            if (is_active[n] == 0 and cur_value(n) is not None
                    and cur_value(n) > max(candidate_value, cur_bid)):
                candidate_value = bidders[n].values[cur_value_idx[n]]
                candidate_bidder = n
        return candidate_value, candidate_bidder

    while True:
        # compute bidding set
        max_inactive_value, entering_bidder = next_candidate()
        while entering_bidder >= 0:
            active_values = [cur_value(j) for j in range(len(bidders)) if is_active[j] == 1]
            if ((sum(is_active) < 2 or
                 h(cur_bid, cur_value(entering_bidder), active_values) >= 0) and
                    not max_inactive_value > cur_bid > max_inactive_value - 1e-8):
                is_active[entering_bidder] = 1
                bid_start_points[entering_bidder][cur_value_idx[entering_bidder]] = cur_bid
                remaining_prob[entering_bidder] = bidders[entering_bidder].prob[cur_value_idx[entering_bidder]]
                max_inactive_value, entering_bidder = next_candidate()
            else:
                break

        # terminates computation
        if sum(is_active) < 2:
            for i in range(len(bidders)):
                if is_active[i] == 1:
                    bid_end_points[i][cur_value_idx[i]] = cur_bid
            break

        # compute next change point
        exiting_criteria = H
        exp_exiting_criteria = exp_H
        change_points = [-1e8] * len(bidders)
        active_values = [cur_value(j) for j in range(len(bidders)) if is_active[j] == 1 and cur_value(j) is not None]
        for i in range(len(bidders)):
            if cur_value(i) is None:
                continue
            if is_active[i] == 0:
                try:
                    change_points[i] = optimize.brentq(
                        lambda x: h(x, cur_value(i), active_values, poly_form=True),
                        -1e8,
                        cur_bid)
                except ValueError:
                    change_points[i] = -1e8
            else:
                if sum(bidders[i].prob[:cur_value_idx[i]]) == 0:
                    change_points[i] = -1e8
                else:
                    try:
                        change_points[i] = optimize.brentq(
                            lambda x: (exiting_criteria(cur_bid, cur_value(i), active_values) -
                                       exiting_criteria(x, cur_value(i), active_values) -
                                       (np.log(sum(bidders[i].prob[:cur_value_idx[i]]) + remaining_prob[i]) -
                                        np.log(sum(bidders[i].prob[:cur_value_idx[i]])))),
                            -1e8,
                            cur_bid)
                    except ValueError:
                        change_points[i] = -1e8

        # update state
        next_change = max(change_points)
        changing_bidder = np.argmax(change_points)
        for i in range(len(bidders)):
            if i == changing_bidder:
                continue
            if is_active[i] == 1:
                remaining_prob[i] = ((sum(bidders[i].prob[:cur_value_idx[i]]) + remaining_prob[i]) /
                                     exp_exiting_criteria(cur_bid, cur_value(i), active_values) *
                                     exp_exiting_criteria(next_change, cur_value(i), active_values) -
                                     sum(bidders[i].prob[:cur_value_idx[i]]))
                if np.abs(remaining_prob[i]) <= 1e-8:
                    F_jump_points[i].append((next_change, sum(bidders[i].prob[:cur_value_idx[i]])))
                    is_active[i] = 0
                    remaining_prob[i] = -1.0
                    bid_end_points[i][cur_value_idx[i]] = next_change
                    cur_value_idx[i] -= 1
                else:
                    F_jump_points[i].append((next_change, sum(bidders[i].prob[:cur_value_idx[i]]) + remaining_prob[i]))
            else:
                F_jump_points[i].append((next_change, sum(bidders[i].prob[:cur_value_idx[i] + 1])))
        if is_active[changing_bidder] == 0:  # entering the bidding set
            is_active[changing_bidder] = 1
            remaining_prob[changing_bidder] = bidders[entering_bidder].prob[cur_value_idx[entering_bidder]]
            bid_start_points[changing_bidder][cur_value_idx[changing_bidder]] = next_change
            F_jump_points[changing_bidder].append(
                (next_change, sum(bidders[changing_bidder].prob[:cur_value_idx[changing_bidder] + 1])))
        else:  # exiting the bidding set
            is_active[changing_bidder] = 0
            remaining_prob[changing_bidder] = -1.0
            bid_end_points[changing_bidder][cur_value_idx[changing_bidder]] = next_change
            cur_value_idx[changing_bidder] -= 1
            F_jump_points[changing_bidder].append(
                (next_change, sum(bidders[changing_bidder].prob[:cur_value_idx[changing_bidder] + 1])))
        cur_bid = next_change
        if cur_bid <= 0.0:
            break
    solution_state = State(is_active=is_active,
                           remaining_prob=remaining_prob,
                           cur_bid=cur_bid,
                           cur_value_idx=cur_value_idx)

    return bid_start_points, bid_end_points, F_jump_points, solution_state


def compute_bidding_stategies(bidders, anonymous_reserve, max_iter=200, tol=1e-6):
    """Compute bidding strategies by iteratively guessing the max winning bid.

    Main arguments:
    max_iter: max number of guessing.
    tol: tolerance of min winning bid: abs(actual min winning bid - guessed min winning bid).
    """
    min_winning_bid = max(compute_min_winning_bid(bidders), anonymous_reserve)

    possible_max_bid_range = [min_winning_bid, max(x.values[-1] for x in bidders)]

    bid_start_points = [[possible_max_bid_range[0]] * len(bidder.values) for bidder in bidders]
    bid_end_points = [[possible_max_bid_range[0]] * len(bidder.values) for bidder in bidders]
    F_jump_points = [[(possible_max_bid_range[0], 1.0)] for _ in range(len(bidders))]

    iter_cnt = 0
    while True:
        iter_cnt += 1
        if iter_cnt > max_iter:
            for i in range(len(bidders)):
                bidders[i].strategy.update(None, None, [])
        cur_guess = (possible_max_bid_range[1] + possible_max_bid_range[0]) / 2.0
        bid_start_points, bid_end_points, F_jump_points, solution_state = (
            compute_strategies_given_max_winning_bid(cur_guess, bidders))
        if np.abs(solution_state.cur_bid - min_winning_bid) < tol:
            break
        if solution_state.cur_bid < min_winning_bid - tol:
            possible_max_bid_range[0] = cur_guess
        else:
            possible_max_bid_range[1] = cur_guess

    for i in range(len(bidders)):
        bidders[i].strategy.update(bid_start_points[i], bid_end_points[i], F_jump_points[i])


def prob_dist(bidders, bid):
    """
    Returns CDF and PDF of each bidder simultaneously.
    For PDF, the point mass at the min winning bid is not considered.
    Returns a list of F(bid) and a list of f(bid)
    """
    assert bid >= 0, 'Bid is negative.'

    active_values = []

    if bid < bidders[0].strategy.F_jump_points[0][0]:
        return [0.0] * len(bidders), [0.0] * len(bidders), active_values
    elif bid >= bidders[0].strategy.F_jump_points[-1][0]:
        return [1.0] * len(bidders), [0.0] * len(bidders), active_values

    # find the smallest jump point that is larger than bid
    prev_jump_idx = -1
    for i in range(len(bidders[0].strategy.F_jump_points)):
        if bidders[0].strategy.F_jump_points[i][0] > bid:
            prev_jump_idx = i
            break

    cdf = [-1.0] * len(bidders)
    pdf = [-1.0] * len(bidders)

    for bidder in bidders:
        for j in range(len(bidder.values)):
            if bidder.strategy.start_points[j] > bid >= bidder.strategy.end_points[j]:
                active_values.append(bidder.values[j])
                break
    for i in range(len(bidders)):
        cur_bidder = bidders[i]
        cur_value = -1.0

        prev_jump_point, prev_jump_F = cur_bidder.strategy.F_jump_points[prev_jump_idx]
        next_jump_point, next_jump_F = cur_bidder.strategy.F_jump_points[prev_jump_idx - 1]

        # find the corresponding value
        for j in range(len(cur_bidder.values)):
            if cur_bidder.strategy.start_points[j] > bid >= cur_bidder.strategy.end_points[j]:
                cur_value = cur_bidder.values[j]
                break

        if cur_value < 0:
            # bid is inactive
            cdf[i] = prev_jump_F
            pdf[i] = 0.0
        else:
            # bid is active
            if bid < (prev_jump_point + next_jump_point) / 2.0:
                cdf[i] = (prev_jump_F / exp_H(prev_jump_point, cur_value, active_values) *
                          exp_H(bid, cur_value, active_values))
            else:
                cdf[i] = (next_jump_F / exp_H(next_jump_point, cur_value, active_values) *
                          exp_H(bid, cur_value, active_values))
            pdf[i] = cdf[i] * h(bid, cur_value, active_values)
    return cdf, pdf, active_values


def compute_fpa_revenue(bidders, anonymous_reserve, points=1000):
    """Computes the revenue of the first price auction.

    Main arguments:
    points: number of points for numerical integration, larger gives higher accuracies.
    """
    rev = 0.0
    bid_range_start = bidders[0].strategy.F_jump_points[0][0]
    bid_range_end = bidders[0].strategy.F_jump_points[-1][0]
    if bid_range_start < bid_range_end:
        bid_range = np.linspace(bidders[0].strategy.F_jump_points[0][0], bidders[0].strategy.F_jump_points[-1][0],
                                num=points)
        bid_range = bid_range[1:]

        rev_b = [0.0] * len(bid_range)
        for j in range(len(bid_range)):
            cdf, _, active_values = prob_dist(bidders, bid_range[j])
            if len(active_values) <= 1:
                rev_b[j] = 0.0
            else:
                rev_b[j] = (bid_range[j] * np.prod(cdf) / (len(active_values) - 1.0) *
                              sum(1.0 / (np.array(active_values) - bid_range[j])))
        rev = integrate.simps(rev_b, bid_range)

    min_winning_bid = max(compute_min_winning_bid(bidders), anonymous_reserve)
    no_winner_prob = 1.0
    for bidder in bidders:
        no_winner_prob *= sum([bidder.prob[j] for j in range(len(bidder.prob)) if bidder.values[j] < min_winning_bid])
    return rev + min_winning_bid * (np.prod([bidder.strategy.F_jump_points[0][1] for bidder in bidders]) -
                                    no_winner_prob)


def compute_fpa_welfare(bidders, anonymous_reserve, points=1000):
    """Compute the welfare of the first price auction

    Main arguments:
    points: number of points for numerical integration, larger gives higher accuracies.
    """
    wel = 0.0
    for i in range(len(bidders[0].strategy.F_jump_points) - 1):
        bid_range_start = bidders[0].strategy.F_jump_points[i][0]
        bid_range_end = bidders[0].strategy.F_jump_points[i + 1][0]
        if bid_range_start < bid_range_end:
            num_points = max(100, int(points / (
                    bidders[0].strategy.F_jump_points[-1][0] - bidders[0].strategy.F_jump_points[0][0]) *
                             (bidders[0].strategy.F_jump_points[i + 1][0] - bidders[0].strategy.F_jump_points[i][0])))
            bid_range = np.linspace(bidders[0].strategy.F_jump_points[i][0],
                                    bidders[0].strategy.F_jump_points[i + 1][0],
                                    num=num_points)
            bid_range = bid_range[1:]

            wel_b = [0.0] * len(bid_range)
            for j in range(len(bid_range)):
                cdf, _, active_values = prob_dist(bidders, bid_range[j])
                if len(active_values) <= 1:
                    wel_b[j] = 0.0
                else:
                    wel_b[j] = np.prod(cdf) * (sum(active_values) / (len(active_values) - 1.0) *
                                               sum(1.0 / (np.array(active_values) - bid_range[j])) -
                                               sum(np.array(active_values) / (np.array(active_values) - bid_range[j])))
            wel += integrate.simps(wel_b, bid_range)

    # point mass at min winning bid
    min_winning_bid = max(compute_min_winning_bid(bidders), anonymous_reserve)
    values = [[-1.0] for _ in range(len(bidders))]
    probs = [[0.0] for _ in range(len(bidders))]
    for i in range(len(bidders)):
        bidder = bidders[i]
        F_min_winning_bid = bidder.strategy.F_jump_points[0][1]
        for j in range(len(bidder.prob)):
            if bidder.values[j] < min_winning_bid:
                probs[i][0] += bidder.prob[j]
            elif sum(bidder.prob[:j]) < F_min_winning_bid:
                values[i].append(bidder.values[j])
                probs[i].append(min(bidder.prob[j], F_min_winning_bid - sum(bidder.prob[:j])))
            else:
                break

    for i in range(len(bidders)):
        for i_value_idx in range(1, len(values[i])):
            i_value = values[i][i_value_idx]
            if i_value == 0.0:
                continue
            winning_prob = probs[i][i_value_idx]
            for j in range(len(bidders)):
                if j == i:
                    continue
                j_losing_prob = 0.0
                for j_value_idx in range(len(values[j])):
                    j_value = values[j][j_value_idx]
                    if j_value < i_value or (j_value == i_value and j < i):
                        j_losing_prob += probs[j][j_value_idx]
                winning_prob *= j_losing_prob
            wel += winning_prob * i_value
    return wel


def compute_spa_revenue(bidders, anonymous_reserve):
    """Computes the revenue of the second price auction"""
    revenue = 0.0
    values = [np.array(bidder.values) for bidder in bidders]
    probs = [np.array(bidder.prob) for bidder in bidders]
    # enumerate the top two bids
    for i in range(len(bidders)):
        for j in range(i + 1, len(bidders)):
            for i_value_idx in range(len(bidders[i].values)):
                if values[i][i_value_idx] < anonymous_reserve:
                    continue
                for j_value_idx in range(len(bidders[j].values)):
                    if values[j][j_value_idx] < anonymous_reserve:
                        continue
                    if values[i][i_value_idx] <= values[j][j_value_idx]:
                        # break ties lexicographically j > i, so j is the winner
                        second_bid = values[i][i_value_idx]
                        second_bidder = i
                    else:
                        second_bid = values[j][j_value_idx]
                        second_bidder = j

                    p = probs[i][i_value_idx] * probs[j][j_value_idx]
                    for k in range(len(bidders)):
                        if p == 0.0:
                            break
                        if k != i and k != j:
                            # k should have a bid smaller than both i and j
                            losing_prob = []
                            for n in range(len(values[k])):
                                if values[k][n] < second_bid or (values[k][n] == second_bid and k < second_bidder):
                                    losing_prob.append(probs[k][n])
                            if len(losing_prob) == 0:
                                p = 0.0
                                break
                            else:
                                p *= sum(losing_prob)
                    revenue += p * second_bid
    # only one bid is above reserve
    for i in range(len(bidders)):
        for i_value_idx in range(len(bidders[i].values)):
            if values[i][i_value_idx] < anonymous_reserve:
                continue
            p = probs[i][i_value_idx]
            for j in range(len(bidders)):
                if j == i:
                    continue
                j_idx = values[j] < anonymous_reserve
                p *= sum(probs[j][j_idx])
            revenue += p * anonymous_reserve

    return revenue


def compute_spa_welfare(bidders, anonymous_reserve):
    wel = 0.0
    for i in range(len(bidders)):
        for i_value_idx in range(len(bidders[i].values)):
            if bidders[i].values[i_value_idx] < anonymous_reserve:
                continue
            p = bidders[i].prob[i_value_idx]
            for j in range(len(bidders)):
                if j == i:
                    continue
                losing_prob = []
                for j_value_idx in range(len(bidders[j].values)):
                    if (bidders[j].values[j_value_idx] < bidders[i].values[i_value_idx] or
                            (bidders[j].values[j_value_idx] == bidders[i].values[i_value_idx] and
                             j < i)):
                        losing_prob.append(bidders[j].prob[j_value_idx])
                p *= sum(losing_prob)
            wel += p * bidders[i].values[i_value_idx]
    return wel
