# Libraries and Classes
import numpy as np
import random
from Agent import Agent
from VDControl import VDControl
import time
import matplotlib.pyplot as plt
from pursuer_controller import pursuer_controller
from evader_controller import evader_controller

# Function for distance calculation
def distance_between(state1, state2):
    d = np.sqrt((state1[0]-state2[0])**2 + (state1[1]-state2[1])**2)
    return d
# function to plot their path
def plot_paths():
    x_e = [0] * (len(deer.controller.path))
    y_e = [0] * (len(deer.controller.path))
    x_p1 = [0] * (len(lassie.controller.path))
    y_p1 = [0] * (len(lassie.controller.path))
    x_p2 = [0] * (len(rex.controller.path))
    y_p2 = [0] * (len(rex.controller.path))
    print(len(deer.controller.path))
    for i in range(len(deer.controller.path)):
        x_e[i] = deer.controller.path[i][0]
        y_e[i] = deer.controller.path[i][1]
        x_p2[i] = rex.controller.path[i][0]
        y_p2[i] = rex.controller.path[i][1]
        x_p1[i] = lassie.controller.path[i][0]
        y_p1[i] = lassie.controller.path[i][1]
    # plt.clf()
    fig, ax = plt.subplots()
    ax.plot(x_p2, y_p2, label='rex')
    plt.plot(x_p1, y_p1, label = 'lassie')
    plt.plot(x_e, y_e, label = 'prey')
    circle = plt.Circle((deer.controller.state[0], deer.controller.state[1]),
                        rex.controller.r, color='g', fill=False)
    circle2 = plt.Circle((deer.controller.state[0], deer.controller.state[1]),
                        3*rex.controller.r, color='y', fill=False)
    plt.plot(deer.controller.state[0], deer.controller.state[1], 'ro')
    ax.add_patch(circle)
    ax.add_patch(circle2)
    plt.legend()
    plt.show()


lassie_counter= 0
rex_counter = 0
deer_counter = 0
capture_region_counter=0
########## TRAINING SECTION ###############
# two agents: lassie and rex are the preditors
# deer is the prey/evader

lassie_pos = [8.5,6]
rex_pos = [7.5,12]
deer_pos = [6,9]
pos_array = [0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9,9.5, 10]

initial_dist_lassie_deer = distance_between(lassie_pos, deer_pos)
initial_dist_rex_deer = distance_between(rex_pos, deer_pos)
print(initial_dist_rex_deer)
print(initial_dist_lassie_deer)

start = time.time() # used to see how long the training time took
lassie_FACLcontroller = pursuer_controller([lassie_pos[0], lassie_pos[1],initial_dist_lassie_deer, initial_dist_rex_deer], [50,50,50,50], [-50,-50,0,0], [20,20,20,20],deer_pos) #create the FACL controller
rex_FACLcontroller = pursuer_controller([rex_pos[0],rex_pos[1],initial_dist_rex_deer, initial_dist_lassie_deer], [50,50,50,50], [-50,-50,0,0],  [20,20,20,20],deer_pos)
deer_FACLcontroller = evader_controller(deer_pos, [50,50], [-50,-50], [7,7])
lassie = Agent(lassie_FACLcontroller) # create the agent with the above controller
rex = Agent(rex_FACLcontroller)
deer = Agent(deer_FACLcontroller)
#print out all the rule sets
#print("rules:")
#print(lassie.controller.rules)

lassie.controller.sigma = 0.3
rex.controller.sigma = 0.3

rolling_success_counter = 0
cycle_counter = 0
for i in range(10000):
    lassie.controller.reset()
    rex.controller.reset()
    deer.controller.reset()
    #choose random positions for lassie and rex
    lassie.controller.state[0] = random.choice(pos_array)
    lassie.controller.state[1] = random.choice(pos_array)
    lassie.controller.path = []
    lassie.controller.path = [lassie.controller.state[0], lassie.controller.state[1]]
    lassie.controller.distance_away_from_target_t = lassie.controller.distance_from_target([0, 0])
    rex.controller.state[0] = random.choice(pos_array)
    rex.controller.state[1] = random.choice(pos_array)
    rex.controller.path = []
    rex.controller.path = [rex.controller.state[0], rex.controller.state[1]]
    rex.controller.distance_away_from_target_t = rex.controller.distance_from_target([0, 0]) #np.sqrt((rex.controller.state[0] - 0)**2 + (rex.controller.state[1] - 0)**2)
    lassie_pos = [lassie.controller.state[0], lassie.controller.state[1]]
    rex_pos = [rex.controller.state[0], rex.controller.state[1]]
    initial_dist_lassie_deer = distance_between(lassie_pos, deer_pos)
    initial_dist_rex_deer = distance_between(rex_pos, deer_pos)

    rex.controller.state[2] = initial_dist_rex_deer
    lassie.controller.state[2] = initial_dist_lassie_deer
    rex.controller.state[3] = initial_dist_lassie_deer
    lassie.controller.state[3] = initial_dist_rex_deer




    cycle_counter += 1
    for j in range(lassie.training_iterations_max):
        # lassie.controller.iterate_train()
        # rex.controller.iterate_train()
        if (lassie.controller.state[3] > lassie.controller.r and rex.controller.state[3] >  rex.controller.r):  ##if both lassie or rex havent captured the deer


            lassie.controller.generate_noise()
            rex.controller.generate_noise()
            deer.controller.generate_noise()

            # Step 3 :  calculate the necessary action
            lassie.controller.calculate_ut()
            rex.controller.calculate_ut()
            deer.controller.calculate_ut()

            # Step 4: calculate the value function at current iterate/time step
            lassie.controller.v_t = lassie.controller.calculate_vt(lassie.controller.phi) #  v_t = sum of self.phi[l] * self.zeta[l]
            rex.controller.v_t = rex.controller.calculate_vt(rex.controller.phi)
            deer.controller.v_t = deer.controller.calculate_vt(deer.controller.phi)

            # Step 5: update the state of the system
            # Update the position in the continuous state space
            #state 0 = x position
            #state 1 = y position
            #state 2 distamce between pursuer and evader
            #state 3 partners distance between pursuer and evader
            lassie.controller.state[0] = lassie.controller.state[0] + lassie.controller.v * np.cos(lassie.controller.u_t) #x pos
            lassie.controller.state[1] = lassie.controller.state[1] + lassie.controller.v * np.sin(lassie.controller.u_t) #y pos
            rex.controller.state[0] = rex.controller.state[0] + rex.controller.v * np.cos(rex.controller.u_t) # x pos
            rex.controller.state[1] = rex.controller.state[1] + rex.controller.v * np.sin(rex.controller.u_t) # y pos
            deer.controller.state[0] = deer.controller.state[0] + deer.controller.v * np.cos(deer.controller.u_t) # x pos
            deer.controller.state[1] = deer.controller.state[1] + deer.controller.v * np.sin(deer.controller.u_t) # y pos

            # Find the distances between the wolfs and the deer
            lassie_pos = [lassie.controller.state[0], lassie.controller.state[1]]
            deer_pos = [deer.controller.state[0], deer.controller.state[1]]
            rex_pos = [rex.controller.state[0], rex.controller.state[1]]
            lassie.controller.state[2] = distance_between(lassie_pos, deer_pos)
            rex.controller.state[2] = distance_between(rex_pos,deer_pos)
            lassie.controller.state[3] = rex.controller.state[2]
            rex.controller.state[3] = lassie.controller.state[2]


            lassie.controller.update_path(lassie_pos)
            lassie.controller.update_input_array(lassie.controller.u_t)
            rex.controller.update_path(rex_pos)
            rex.controller.update_input_array(rex.controller.u_t)
            deer.controller.update_input_array(deer.controller.u_t)
            deer.controller.update_path(deer_pos)

            lassie.controller.phi_next = lassie.controller.update_phi()
            rex.controller.phi_next = rex.controller.update_phi()
            deer.controller.phi_next = deer.controller.update_phi()


            # Step 6: get reward, this will be replaced for value decomp?
            # lassie.controller.reward = lassie.controller.get_reward()
            # lassie.controller.reward = rex.controller.get_reward()
            lassie.controller.distance_away_from_target_t_plus_1 = lassie.controller.distance_from_target(deer_pos) #np.sqrt( (lassie.controller.state[0] - deer.controller.state[0])**2 +(lassie.controller.state[1] - deer.controller.state[1])**2))
            rex.controller.distance_away_from_target_t_plus_1 = rex.controller.distance_from_target(deer_pos) #np.sqrt( (rex.controller.state[0] - deer.controller.state[0])**2 +(rex.controller.state[1] - deer.controller.state[1])**2))
            deer.controller.distance_away_from_target_t_plus_1 = deer.controller.distance_from_target()

            lassie_individual_r = -3
            rex_individual_r = -5
            team_reward = 0
            w = 0.5 # weight of individual reward to team reward

            # TERMINAL REWARDS

            if (rex.controller.state[2] < rex.controller.r):
                rex_individual_r = 5
                rex.success += 1
                if(lassie.controller.state[2] < 3*lassie.controller.r):
                    team_reward = 7
                    capture_region_counter+=1
                lassie.controller.reward = w*lassie_individual_r + (1-w)*(lassie_individual_r+rex_individual_r+team_reward)
                rex.controller.reward = w * rex_individual_r + (1 - w) * (lassie_individual_r + rex_individual_r + team_reward)
                deer.controller.reward = 0
            elif (lassie.controller.state[2] < lassie.controller.r):
                lassie_individual_r = 3
                lassie.success += 1
                # check if rex was in the circle
                if(rex.controller.state[2] < 3*rex.controller.r):
                    team_reward = 7
                    capture_region_counter += 1
                lassie.controller.reward = w * lassie_individual_r + (1 - w) * (
                                lassie_individual_r + rex_individual_r + team_reward)
                rex.controller.reward = w * rex_individual_r + (1 - w) * (
                                lassie_individual_r + rex_individual_r + team_reward)
                deer.controller.reward = 0
            # SHAPING REWARDS
            else:
                lassie.controller.reward = (lassie.controller.distance_away_from_target_t - lassie.controller.distance_away_from_target_t_plus_1)
                rex.controller.reward = (rex.controller.distance_away_from_target_t - rex.controller.distance_away_from_target_t_plus_1)
                deer.controller.reward = (
                            deer.controller.distance_away_from_terr_t - deer.controller.distance_away_from_terr_t_plus_1)

            # print("reward", self.distance_away_from_target_t, '-', self.distance_away_from_target_t_plus_1, '=', r)
            lassie.controller.distance_away_from_target_t = lassie.controller.distance_away_from_target_t_plus_1
            rex.controller.distance_away_from_target_t = rex.controller.distance_away_from_target_t_plus_1
            deer.controller.distance_away_from_terr_t = deer.controller.distance_away_from_terr_t_plus_1

            lassie.controller.update_reward_graph(lassie.controller.reward)
            rex.controller.update_reward_graph(rex.controller.reward)
            deer.controller.update_reward_graph(deer.controller.reward)

            # Step 7: Calculate the expected value for the next step
            lassie.controller.v_t_1 = lassie.controller.calculate_vt(lassie.controller.phi_next) # self.phi[l] * self.zeta[l]
            rex.controller.v_t_1 = rex.controller.calculate_vt(rex.controller.phi_next)
            deer.controller.v_t_1 = deer.controller.calculate_vt(deer.controller.phi_next)

            # Step 8: calculate the temporal difference
            #No VD
            lassie.controller.calculate_prediction_error()
            rex.controller.calculate_prediction_error()
            deer.controller.calculate_prediction_error()


            # Step 9: update the actor and critic functions
            lassie.controller.update_zeta() # update the critic
            lassie.controller.update_omega() # update the actor
            rex.controller.update_zeta()  # update the critic
            rex.controller.update_omega()  # update the actor
            deer.controller.update_zeta()  # update the critic
            deer.controller.update_omega()  # update the actor

            lassie.controller.phi = lassie.controller.phi_next
            rex.controller.phi = rex.controller.phi_next
            deer.controller.phi = deer.controller.phi_next

        else: #if prey was caught
            break


    lassie.controller.updates_after_an_epoch()
    lassie.reward_total.append(lassie.reward_sum_for_a_single_epoch())
    rex.controller.updates_after_an_epoch()
    rex.reward_total.append(rex.reward_sum_for_a_single_epoch())

    # check to see if we should stop training based on a rolling counter
    # if we hit 2k consecutive successful training rounds, then stop training
    # if(rolling_success_counter != cycle_counter):
    #     cycle_counter=0
    #     rolling_success_counter=0
    #
    # if (rolling_success_counter >= 1000):
    #     print('number of epochs trained: ', i)
    #     break
    # print out some stats as it trains every so often
    if (i % 500 == 0):
        print('epoch number : ', i)
        print("time:", time.time()-start)
        print("xy path of lassie",lassie.controller.path[len(lassie.controller.path)-1]) #numerical values of path
        print("xy path of rex", rex.controller.path[len(rex.controller.path)-1])  # numerical values of path
        print("xy path of deer", deer.controller.path[len(deer.controller.path) - 1])
        print('length of game', len(rex.controller.path))
        print('sigma ', rex.controller.sigma)
        print('lassie wins : ', lassie.success)
        print('rex wins ', rex.success)
        print('capture regions ', capture_region_counter)

        #print("input, ut:", lassie.controller.input)

end = time.time()
print('total train time : ', end-start)
print(' total num of successes during training for lassie : ', lassie.success)
print(' total num of successes during training for rex : ', rex.success)

# Print the path that our agent lassie took in her last epoch
#print("xy path",lassie.controller.path) #numerical values of path
# print("input, ut:" , lassie.controller.input)



lassie.success = 0
rex.success = 0
number_of_ties=0
capture_region_counter=0
lassie.controller.sigma = 0.15
rex.controller.sigma = 0.15
#Run a series of games
for i in range(1000):
    lassie.controller.reset()
    rex.controller.reset()
    deer.controller.reset()
    for j in range(lassie.training_iterations_max):
        # lassie.controller.iterate_train()
        # rex.controller.iterate_train()
        if (lassie.controller.state[3] > lassie.controller.r and rex.controller.state[3] > rex.controller.r):
            lassie.controller.generate_noise()
            rex.controller.generate_noise()
            deer.controller.generate_noise()
            # Step 3 :  calculate the necessary action
            lassie.controller.calculate_ut()
            rex.controller.calculate_ut()
            deer.controller.calculate_ut()

            # Step 5: update the state of the system
            lassie.controller.state[0] = lassie.controller.state[0] + lassie.controller.v * np.cos(
                lassie.controller.u_t)  # x pos
            lassie.controller.state[1] = lassie.controller.state[1] + lassie.controller.v * np.sin(
                lassie.controller.u_t)  # y pos
            rex.controller.state[0] = rex.controller.state[0] + rex.controller.v * np.cos(rex.controller.u_t)  # x pos
            rex.controller.state[1] = rex.controller.state[1] + rex.controller.v * np.sin(rex.controller.u_t)  # y pos
            deer.controller.state[0] = deer.controller.state[0] + deer.controller.v * np.cos(
                deer.controller.u_t)  # x pos
            deer.controller.state[1] = deer.controller.state[1] + deer.controller.v * np.sin(
                deer.controller.u_t)  # y pos

            # Find the distances between the wolfs and the deer
            lassie_pos = [lassie.controller.state[0], lassie.controller.state[1]]
            deer_pos = [deer.controller.state[0], deer.controller.state[1]]
            rex_pos = [rex.controller.state[0], rex.controller.state[1]]
            lassie.controller.state[2] = distance_between(lassie_pos, deer_pos)
            rex.controller.state[2] = distance_between(rex_pos, deer_pos)
            lassie.controller.state[3] = rex.controller.state[2]
            rex.controller.state[3] = lassie.controller.state[2]

            lassie.controller.update_path(lassie_pos)
            lassie.controller.update_input_array(lassie.controller.u_t)
            rex.controller.update_path(rex_pos)
            rex.controller.update_input_array(rex.controller.u_t)
            deer.controller.update_input_array(deer.controller.u_t)
            deer.controller.update_path(deer_pos)

            lassie.controller.phi_next = lassie.controller.update_phi()
            rex.controller.phi_next = rex.controller.update_phi()
            deer.controller.phi_next = deer.controller.update_phi()
        else:  # if an agent has crossed the line
            if (rex.controller.state[2] < rex.controller.r):
                rex.success += 1
                if (lassie.controller.state[2] < 3 * lassie.controller.r):
                    capture_region_counter +=1

            elif (lassie.controller.state[2] < lassie.controller.r):
                lassie.success += 1
                # check if rex was in the circle
                if (rex.controller.state[2] < 3 * rex.controller.r):
                    capture_region_counter += 1
            break

print('GAME STATS ')
print('lassie wins : ', lassie.success)
print('rex wins ', rex.success)
print('capture_region_counter ', capture_region_counter)

# lassie.save_epoch_training_info() #save all the important info from our training sesh

plot_paths()

#Print out the reward plots combined
fig, ax = plt.subplots()
plt.title('lassie and rex rewards per epoch')
ax.plot(lassie.reward_total,label='lassie')
ax.plot(rex.reward_total,label='rex')
plt.xlabel('epoch')
plt.ylabel('total rewards per epoch')
plt.legend()
plt.show()
#
# lassie.print_reward_graph()
# rex.print_reward_graph()


