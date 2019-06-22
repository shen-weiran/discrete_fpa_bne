import matplotlib.pyplot as plt
from fpa_bne import *


def main():
    # Init
    bidder_values = [
        [1, 2, 3, 4, 5],
        [0, 1, 3, 4, 5]
    ]

    bidder_probs = [
        [0.2, 0.1, 0.1, 0.2, 0.4],
        [0.4, 0.1, 0.3, 0.1, 0.1]
    ]

    bidders = [
        Bidder(bidder_values[i], bidder_probs[i]) for i in range(len(bidder_values))
    ]

    reserve_price = 0.0

    # Compute bidding strategies
    compute_bidding_stategies(bidders, reserve_price, tol=1e-8)
    min_winning_bid = compute_min_winning_bid(bidders)

    print('Min winning bid: {}'.format(min_winning_bid))
    print('Max winning bid: {}'.format(bidders[0].strategy.F_jump_points[-1][0]))

    fpa_welfare = compute_fpa_welfare(bidders, reserve_price)
    print('Welfare of first price auction: {}'.format(fpa_welfare))

    fpa_revenue = compute_fpa_revenue(bidders, reserve_price)
    print('Revenue of first price auction: {}'.format(fpa_revenue))

    spa_welfare = compute_spa_welfare(bidders, reserve_price)
    print('Welfare of second price auction: {}'.format(spa_welfare))

    spa_revenue = compute_spa_revenue(bidders, reserve_price)
    print('Revenue of second price auction: {}'.format(spa_revenue))

    bid_range = np.linspace(bidders[0].strategy.F_jump_points[0][0],
                            bidders[0].strategy.F_jump_points[-1][0],
                            num=1000)
    bid_range = bid_range[1:]

    # Compute pdf
    all_bidder_pdf = np.zeros([len(bidders), len(bid_range)])
    for j in range(len(bid_range)):
        cdf, pdf, _ = prob_dist(bidders, bid_range[j])
        for i in range(len(bidders)):
            all_bidder_pdf[i][j] = pdf[i]

    # Plot bid distribution
    plt.figure(figsize=(8, 4 * len(bidders)))
    for i in range(len(bidders)):
        plt.subplot(len(bidders), 1, i + 1)
        plt.plot(bid_range, all_bidder_pdf[i])
        plt.ylabel('Bidder {}'.format(i + 1))
    plt.suptitle('Bid PDF of equilibrium')
    plt.show()


if __name__ == '__main__':
    main()
