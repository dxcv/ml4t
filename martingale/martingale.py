"""Assess a betting strategy.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		   	  			  	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		   	  			  	 		  		  		    	 		 		   		 		  
All Rights Reserved  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		   	  			  	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		   	  			  	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		   	  			  	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		   	  			  	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		   	  			  	 		  		  		    	 		 		   		 		  
or edited.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		   	  			  	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		   	  			  	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		   	  			  	 		  		  		    	 		 		   		 		  
GT honor code violation.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---
Student Name: Christopher Fleisher
GT User ID: cfleisher3
GT ID: 903421975
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import comb

def author():
    return 'cfleisher3'  # replace tb34 with your Georgia Tech username.


def gtid():
    return 900897987  # replace with your GT ID number


def get_spin_result(win_prob):
    result = False
    if np.random.random() <= win_prob:
        result = True
    return result


def run_simulator(win_prob, bank_roll=None):
    # [end_purse, start loss_count, win or loss, wager]
    spins = np.zeros((1000, 4))
    spin_idx = 0

    avail_roll = bank_roll
    if avail_roll is not None:
        avail_roll = abs(avail_roll)

    while True:
        # break after 1,000 spins
        if spin_idx > spins.shape[0] - 1:
            break

        # don't play if purse at least 80 or bank roll zero
        start_purse = spins[max(spin_idx-1, 0)][0]
        if start_purse >= 80 or avail_roll == 0:
            spins[spin_idx][0] = start_purse
            spin_idx += 1
            continue

        loss_cnt = 0
        won = False
        while True:
            # break after 1,000 spins
            if spin_idx > spins.shape[0] - 1:
                break

            won = get_spin_result(win_prob)

            wager = 2**loss_cnt
            if avail_roll is not None:
                wager = min(wager, avail_roll)

            if won:
                trial = np.array([start_purse+wager, loss_cnt, 1., wager])
                spins[spin_idx] = trial
                spin_idx += 1
                if avail_roll is not None:
                    avail_roll = min(bank_roll, avail_roll + wager)
                break
            else:
                start_purse -= wager
                if avail_roll is not None:
                    avail_roll = max(avail_roll - wager, 0)
                trial = np.array([start_purse, loss_cnt, 0., wager])
                spins[spin_idx] = trial
                loss_cnt += 1
                spin_idx += 1

    return spins


def experiment1(win_prob):
    # subplot 1: 10 simulations
    data1 = np.array([run_simulator(win_prob) for _ in range(10)])
    winnings1 = data1[:, :, 0]

    plt.subplot(511)
    for row in winnings1:
        plt.plot(row)

    plt.title('Experiment 1')
    plt.axis([0, 300, -256, 100])
    plt.xlabel('simulation step')
    plt.ylabel('winnings ($)')

    # subplot 2 (mean +/- 1 std)
    data2 = np.array([run_simulator(win_prob) for _ in range(1000)])
    winnings2 = data2[:, :, 0]
    mean_data = np.mean(winnings2, axis=0)
    std_data = np.std(winnings2, axis=0)
    chart2_data = np.array([
        mean_data + std_data,
        mean_data,
        mean_data - std_data,
        ])
    chart2_labels = ['mean+1 std', 'mean', 'mean-1 std']

    plt.subplot(512)
    for i, row in enumerate(chart2_data):
        plt.plot(row, label=chart2_labels[i])

    plt.axis([0, 300, -256, 100])
    plt.xlabel('simulation step')
    plt.ylabel('winnings ($)')
    plt.legend()

    # subplot 3 (median +/- 1 std)
    median_data = np.median(winnings2, axis=0)
    chart3_data = np.array([
        median_data + std_data,
        median_data,
        median_data - std_data,
        ])
    chart3_labels = ['median+1 std', 'median', 'median-1 std']

    plt.subplot(513)
    for i, row in enumerate(chart3_data):
        plt.plot(row, label=chart3_labels[i])

    plt.axis([0, 300, -256, 100])
    plt.xlabel('simulation step')
    plt.ylabel('winnings ($)')
    plt.legend()


def experiment2(win_prob):
    # subplot 4: 1000 simulations with mean +/- 1 std
    sims = range(1000)
    data = np.array([run_simulator(win_prob, bank_roll=256) for _ in sims])
    winnings = data[:, :, 0]

    mean_data = np.mean(winnings, axis=0)
    std_data = np.std(winnings, axis=0)
    chart4_data = np.array([
        mean_data + std_data,
        mean_data,
        mean_data - std_data,
        ])
    chart4_labels = ['mean+1 std', 'mean', 'mean-1 std']

    plt.subplot(514)
    for i, row in enumerate(chart4_data):
        plt.plot(row, label=chart4_labels[i])

    plt.axis([0, 300, -256, 100])
    plt.xlabel('simulation step')
    plt.ylabel('winnings ($)')
    plt.legend()

    median_data = np.median(winnings, axis=0)
    chart5_data = np.array([
        median_data + std_data,
        median_data,
        median_data - std_data,
        ])
    chart5_labels = ['median+1 std', 'median', 'median-1 std']

    plt.subplot(515)
    for i, row in enumerate(chart5_data):
        plt.plot(row, label=chart5_labels[i])

    plt.axis([0, 300, -256, 100])
    plt.xlabel('simulation step')
    plt.ylabel('winnings ($)')
    plt.legend()


def peak_drawdown(bets):
    drawdown = 0
    print(f'bet\tloss\tdrawdown')
    for i in range(1, bets):
        loss = 2**(i-1)
        drawdown -= loss
        print(f'{i}\t{loss}\t{drawdown}')
    return drawdown


def calc_prob(max_bets, net_wins):
    p = 18/38
    q = 1 - p
    max_loss = max_bets-net_wins+1
    probs = np.zeros(max_bets-net_wins+1)
    probs[0] = p**net_wins
    for i in range(1, probs.shape[0]):
        prior_probs = 1 - np.sum(probs[:i])

        prior_total = net_wins+i
        prior_wins = net_wins + (i-1)//2
        prior_losses = prior_total - prior_wins
        # print(f'prior total: {prior_total}, prior wins: {prior_wins}, prior_losses: {prior_losses}')

        prior_combo = comb(prior_total, prior_wins)
        prior_exp_prob = (p**prior_wins)*(q**prior_losses)

        cur_prob = p
        if prior_total % 2 == 0:
            cur_prob *= p

        probs[i] = prior_probs*prior_combo*prior_exp_prob*cur_prob

        # prior_probs = 1 - np.sum(probs[:i])
        # prior_bets = 2*(i-1)+net_wins
        # prior_wins = net_wins-1
        # prior_combo = comb(prior_bets, prior_wins)

        # cur_wins = prior_wins+2
        # cur_losses = prior_bets+2-cur_wins
        # cur_prob = (p**(cur_wins))*q**(cur_losses)
        # # print(f'prior_probs: {prior_probs}; prior_combo: {prior_combo}; cur_prob: {cur_prob}')
        # probs[i] = prior_probs*prior_combo*cur_prob
        # print(f'prior_probs:{prior_probs}; prior_combo:{prior_combo}; cur_prob:{cur_prob}')
        # print(f'prior bets: {prior_bets}, prior wins: {prior_wins}, cur_wins: {cur_wins}, cur_losses: {cur_losses}')
        # print(f'i 0: {probs[i]}')
    return probs


def experiment1_prob(win_prob, max_bets, net_win):
    # P(X>=80)= 1 - P(X<80)
    loss_prob = 1 - win_prob
    total = 1.0
    for wins in range(80):
        losses = max_bets - wins
        total -= comb(max_bets, wins)*(win_prob**wins)*loss_prob**losses
    return total


def experiment1_exp_val(win_prob, n):
    loss_prob = 1.0 - win_prob
    ev = 0.0
    for i in range(n+1):
        ev += win_prob - loss_prob

    return ev


def test_code():
    win_prob = 18 / 38  # american roulette: 18 black, 18 red, 2 green
    np.random.seed(gtid())  # do this only once

    # end_purse, start loss_count, win or loss, wager
    # experiment1(win_prob)
    # experiment2(win_prob)
    # plt.show()

    max_bets = 1000
    net_wins = 80
    result = experiment1_prob(win_prob, max_bets, net_wins)
    print(f'sum: {result}')

    ev = experiment1_exp_val(win_prob, 1000)
    print(f'ev: {ev}')


if __name__ == "__main__":
    test_code()
