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


def author():
    return 'cfleisher3'  # replace tb34 with your Georgia Tech username.


def gtid():
    return 900897987  # replace with your GT ID number


def get_spin_result(win_prob):
    result = False
    if np.random.random() <= win_prob:
        result = True
    return result


def run_simulator(win_prob):
    spins = np.zeros((1000, 4))
    spin_idx = 0
    while True:
        # break after 1,000 spins
        if spin_idx > spins.shape[0] - 1:
            break

        # don't play if purse at least 80
        start_purse = spins[max(spin_idx-1, 0)][0]
        if start_purse >= 80:
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
            if won:
                trial = np.array([start_purse+wager, loss_cnt, 1., wager])
                spins[spin_idx] = trial
                spin_idx += 1
                break
            else:
                start_purse -= wager
                trial = np.array([start_purse, loss_cnt, 0., wager])
                spins[spin_idx] = trial
                loss_cnt += 1
                spin_idx += 1

    return spins


def experiment1(win_prob):
    # subplot 1: 10 simulations
    data1 = np.array([run_simulator(win_prob) for _ in range(10)])
    winnings1 = data1[:, :, 0]

    plt.subplot(311)
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

    plt.subplot(312)
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

    plt.subplot(313)
    for i, row in enumerate(chart3_data):
        plt.plot(row, label=chart3_labels[i])

    plt.axis([0, 300, -256, 100])
    plt.xlabel('simulation step')
    plt.ylabel('winnings ($)')
    plt.legend()

    plt.show()


def test_code():
    win_prob = 18 / 38  # american roulette: 18 black, 18 red, 2 green
    np.random.seed(gtid())  # do this only once

    # end_purse, start loss_count, win or loss, wager
    experiment1(win_prob)


if __name__ == "__main__":
    test_code()
