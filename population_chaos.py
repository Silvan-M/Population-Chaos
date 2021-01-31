
import random
from itertools import count
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from copy import deepcopy
import math

class MultiplePopulations:
	def __init__ (self, n, name='winter', pandemic = False, starting_population=10): 
		# n = amount of simulated populations, creature = class object, name = for colormapping
		cmap = plt.cm.get_cmap(name, n+1)
		self.multiple_population = []
		if pandemic: 
			for i in range(n):
				self.multiple_population.append(Creature(color=cmap(i),
												birth_rate=0,
												death_rate=0,
												reproduction_rate=0, 
												mutations_dict={0:0}, 
												start_amount=starting_population, 
												pandemic = True, 
												wears_mask_prob = 0, 
												moving_prob = 1, 
												position = [10,10], 
												max_distance = 0.2, 
												prob_cont = 0.2, 
												prob_mask_red = 0.0, 
												is_dead = False, 
												prob_death = 0.01, 
												initial_prob = 0.01,	 
												hotspot = [5,5],
												hotspot_prob = 1,
												hotspot_sigma = 2))
		else:
			for i in range(n):
				self.multiple_population.append(Creature(color=cmap(i),birth_rate=0,death_rate=0.5,reproduction_rate=0.5, mutations_dict={0:0}, start_amount=100))

class Creature:
		def __init__(self, color, birth_rate, death_rate, reproduction_rate, start_amount=0, mutations_dict={}, pandemic = False, wears_mask_prob = 0, moving_prob = 0, position = [10,10], max_distance = 0, prob_cont = 0.3, prob_mask_red = 0.02, is_dead = False, prob_death = 0.001, initial_prob = 0.1, hotspot = False, hotspot_prob = 0.1, hotspot_sigma = 2):
				self.color = color	# Needs to be unique
				self.birth_rate = birth_rate
				self.death_rate = death_rate
				self.reproduction_rate = reproduction_rate
				self.mutations_dict = mutations_dict	# Format: {index:rate}
				self.start_amount = start_amount
				self.history = [start_amount]
				self.type = 0
				self.is_dead = is_dead
				self.hotspot = hotspot # Format: Boolean or [x,y]
				self.hotspot_prob = hotspot_prob # Format: [0,1]
				self.hotspot_sigma = hotspot_sigma


				# Pandemic statsb, boolean:
				if pandemic:
					self.pandemic = True
					# Sets chance of contaminating others
					self.prob_cont = prob_cont
					# Sets reduction of the probability by wearing mask
					self.prob_mask_red = prob_mask_red
					# Sets chance of dying due to infection
					self.prob_death = prob_death
					# Decides if creatures wears a mask
					if wears_mask_prob >= random.random():
						self.wears_mask = True
					else:
						self.wears_mask = False
					# Decides if creature is moving
					if moving_prob >= random.random():
						self.is_moving = True
					else:
						self.is_moving = False
					# Chooses initial suspects
					self.initial_prob = initial_prob
				
					if initial_prob >= random.random():
						print('w')
						self.is_contaminated = True
					else:
						self.is_contaminated = False

					# Sets range of field [x,y]
					self.range_position = position
					# Sets creature to random position within range
					self.position = [random.randint(0,position[0]),random.randint(0,position[1])]
					
					# Counts time since infection [generations]
					self.time_since_infection = 0
					# If creature ca infect others
					self.can_infect = True
					# If creature is immune
					self.is_immune = False
					# Max distance where infections can take place
					self.max_distance = max_distance

					# MOVEMENT VARIABLES
					# Angle of next movement direction starting from the up at 0 clockwise up until 2*pi which is again up
					self.angle = 0 
					# Velocity the agent is moving
					self.velocity = 0.05
					# Maximum velocity the creature can walk
					self.max_velocity = 1
				else:
					self.pandemic = False

		def reset(self):
			if self.initial_prob >= random.random():
						print('w')
						self.is_contaminated = True
			self.position = [random.randint(0,self.range_position[0]),random.randint(0,self.range_position[1])]


		def _randomize_angle_velocity(self):
			'''
			Changes angle and velocity randomly.
			'''

			# Increase or decrease velocity by [-0.15,0.15]
			self.velocity += random.uniform(-0.15,0.15)
			if self.velocity >= self.max_velocity:
				# If new velocity exceeds maximum velocity change velocity to maximum velocity
				self.velocity = self.max_velocity
			elif self.velocity < 0:
				# If new velocity is below 0 change velocity to 0 
				self.velocity = 0
			#print(f'pos;{self.position}',end=' ')
			if self.hotspot != False:
				if self.hotspot_prob >= random.random():
					# Calculate direct vector to hotspot and unitize it
					direction_vector = [(self.hotspot[0]-self.position[0]),(self.hotspot[1]-self.position[1])]
					if direction_vector[0] < 0 and direction_vector[1] < 0:
						quadrant = 3
					elif direction_vector[0] < 0 and direction_vector[1] > 0:
						quadrant = 2
					elif direction_vector[0] > 0 and direction_vector[1] < 0:
						quadrant = 4
					else:
						quadrant = 1


					length = np.sqrt(np.square(direction_vector[0])+np.square(direction_vector[1]))
					unit_direction_vector = [direction_vector[0]/length, direction_vector[1]/length]
					print(f'unit vector;{direction_vector}',end = ' ')
					# Calculate according angle 
					try:
						x = unit_direction_vector[0]
						y = unit_direction_vector[1]
						direction_angle = np.arctan(y/x)
					except ZeroDivisionError:
						if self.position[0] > self.hotspot[0]:
							direction_angle = np.pi
						else:
							direction_angle = 0

					
					if quadrant == 1: 
						direction_angle = (1/2)*np.pi - direction_angle
					elif quadrant == 2: 
						direction_angle =	direction_angle
					elif quadrant == 3:	
						direction_angle = (np.pi) + direction_angle
					elif quadrant == 4:
						direction_angle = (1/2)*(np.pi) - direction_angle

					print(f'direction_angle;{direction_angle}',end = ' ')
					print(f'1self.angle;{self.angle}',end = ' ')
					# Calculate the change of angle needed to go into hotspot direction
					delta_angle = self.angle - (direction_angle) 
					print(f'delta;{delta_angle}',end = ' ')
					# Set the	mean and standard deviation for the normal distribution
					mu, sigma = delta_angle, self.hotspot_sigma
					# make a choice according to the normal distribution 

					angle = np.random.normal(mu, sigma, 1)



					print(f'choosen angle;{angle}',end = ' ')
					self.angle -= delta_angle
					print(f'new angle;{self.angle}\n')
					if self.angle > 2*np.pi:
						# If new angle exceeds 1 reset angle to be in range [0,1]
						self.angle -= 2*np.pi
					elif self.angle < 0:
						# If new angle is below 0 reset angle to be in range [0,1]
						self.angle += 2*np.pi
				
			
			else:
				# Increase or decrease angle by [-0.15,0.15]
				self.angle += random.uniform(-0.15,0.15)
				if self.angle > 1:
					# If new angle exceeds 1 reset angle to be in range [0,1]
					self.angle -= 1
				elif self.angle < 0:
					# If new angle is below 0 reset angle to be in range [0,1]
					self.angle += 1

			


			

		def __str__(self):
				return "Color: {}\nAmount: {}\n".format(self.color, self.amount)

		def set_type(self, type):
				'''
				Sets the type of the creature which consists of the index of the list in creatures_types
				'''
				self.type = type


class Environment:
	def __init__(self, creatures, observers=[], pandemic=False, size=10, movement_mode=0, boundaries=True):
		self.creatures = []
		self.creature_types = creatures
		self.plots = []
		self.observers = observers
		self.pandemic = pandemic
		self.size = size
		self.movement_mode = movement_mode
		self.boundaries = boundaries

		if pandemic:
			self.pandemic_history = [[],[],[]]
			for main_creature in (self.creatures):
				if main_creature.is_moving and main_creature.is_dead == False:
						# Sets new random position
						main_creature.position = [random.randint(0,main_creature.range_position[0]),random.randint(0,main_creature.range_position[1])]

		# Creating plots list for population counting
		for i, creature_type in enumerate(self.creature_types):
				self.plots.append([creature_type.color,list()])
				creature_type.set_type(i)
				for v in range(creature_type.start_amount):
					deep = deepcopy(creature_type)
					deep.reset()
					self.creatures.append(deep)

	def step(self):
		'''
		Advances simulation one timestep and returns all creatures as objects.
		'''
		for creature_type in self.creature_types:
				self.plots[creature_type.type][1].append(0)
				# A random creature gets born
				if creature_type.birth_rate >= random.random():
						self.creatures.append(deepcopy(creature_type))

		deaths = 0
		creatures_copy = self.creatures.copy()
		for i, creature in enumerate(self.creatures):
				# The n-th creature dies
				if (not creature.is_dead) and creature.death_rate >= random.random():
					creatures_copy[i] = 0
					deaths += 1

		self.creatures = [creature for creature in creatures_copy if creature != 0]

		# print("There are {} creatures! {} creatures died! That's {:1.3f}%.".format(len(self.creatures),deaths,100/(len(self.creatures)+deaths)*deaths))

		# Survivors
		for creature in self.creatures:
			if not creature.is_dead:
				self.plots[creature.type][1][-1] += 1

				# The n-th creature reproduces
				if creature.reproduction_rate >= random.random():
						# Check if the n-th creature mutates while reproducing and if so into which color it mutates
						mutation = -1

						for mutation_key in creature.mutations_dict:
								if creature.mutations_dict[mutation_key] >= random.random():
										mutation = mutation_key

								if mutation == -1:
										# No mutation occurs
										self.creatures.append(self.creature_types[creature.type])
								else:
										# A mutation occurs
										self.creatures.append(self.creature_types[mutation])


					# If pandemic is switched on
		if self.pandemic:
			# Iterate over all creatures
			for main_creature in (self.creatures):
				# Checks if creature is dead (cannot contaminate others)
				if main_creature.is_dead == False:
					# Checks if creature is contaminated and can therefore infect others
					if main_creature.is_contaminated:
						# Increases counter by one if creature is contaminated
						main_creature.time_since_infection += 1
						# Checks every other creature
						for sub_creature in (self.creatures):
							
							# Calculates the distance to that creature, if smaller	than the max contaminating range
							dist = self.get_distance(main_creature, sub_creature)
							if dist	<= main_creature.max_distance:
								
								# Set chance of an infection to standard probability, decreasing with distance^-2
								chance = 0
								if dist == 0:
									chance = 1
								else:
									chance = main_creature.prob_cont * (1/np.square(dist))
								# Reduces chance if one or both wear masks
								if main_creature.wears_mask:
									chance = chance - main_creature.prob_mask_red
								if sub_creature.wears_mask:
									chance = chance - sub_creature.prob_mask_red
								# Defines if the creature gets infected
								if chance >= random.random():
									sub_creature.is_contaminated = True

			amount_healthy = 0
			amount_contaminated = 0
			amount_dead = 0

			for main_creature in (self.creatures):
				if main_creature.is_contaminated:
					chance = self.get_death_prob()* main_creature.prob_death
					if chance >= random.random():
						main_creature.is_dead = True

				if main_creature.is_dead:
					amount_dead += 1
				elif main_creature.is_contaminated:
					amount_contaminated += 1
				else:
					amount_healthy += 1

				# If creature is allowed to move
				if main_creature.is_moving and main_creature.is_dead == False:
					# MOVEMENT MODE [0: teleport, 1: linear random movement, 2: linear random movement and hotspots]
					if self.movement_mode == 0:
						# TELEPORT
						# Sets new random position
						pos = [random.uniform(0,main_creature.range_position[0]),random.uniform(0,main_creature.range_position[1])]
						main_creature.position = pos 
					elif self.movement_mode == 1:
						# LINEAR RANODOM MOVEMENT
						main_creature._randomize_angle_velocity()
						main_creature.position[0] += math.sin(main_creature.angle)*main_creature.velocity
						main_creature.position[1] += math.cos(main_creature.angle)*main_creature.velocity
						

						if self.boundaries:
							# Check if position is outside of the defined boundaries, go through x- and y-coordinates and reset if necessary
							for i in range(2):
								if main_creature.position[i] < 0:
									main_creature.position[i] = 0
								elif main_creature.position[i] > self.size:
									main_creature.position[i] = self.size
					else:
						# LINEAR RANODOM MOVEMENT WITH HOTSPOTS

						# HOTSPOTS TBD! #
						main_creature._randomize_angle_velocity()
						main_creature.position[0] += math.sin(main_creature.angle)*main_creature.velocity
						main_creature.position[1] += math.cos(main_creature.angle)*main_creature.velocity

						x = math.sin(main_creature.angle)*main_creature.velocity
						y = math.cos(main_creature.angle)*main_creature.velocity
						length = np.sqrt(np.square(x)+np.square(y))
						x = x/length
						y = y/length
						print(f'vector;{x},{y}',end=' ')


						if self.boundaries:
							# Check if position is outside of the defined boundaries, go through x- and y-coordinates and reset if necessary
							for i in range(2):
								if main_creature.position[i] < 0:
									main_creature.position[i] = 0
								elif main_creature.position[i] > self.size:
									main_creature.position[i] = self.size
			
			print("Amount Contaminated: ",amount_contaminated)
			print("Amount Dead: ",amount_dead)
			print("Amount Healthy: ",amount_healthy)
			self.pandemic_history[0].append(amount_healthy)
			self.pandemic_history[1].append(amount_contaminated)
			self.pandemic_history[2].append(amount_dead)
		for obs in self.observers:
			obs.notify()
	
	# Function which defines how likely it is to die after a certain amount of time; time_since_infection [generations] --> [0,1]
	def get_death_prob(self):
		chance = 1
		return(chance)

	def get_distance(self, creature1, creature2):
		return(np.sqrt(np.square(creature1.position[0]-creature2.position[0])+np.square(creature1.position[1]-creature2.position[1])))

	def get_pandemic_plots(self, last=0):
		'''
		Returns a plot of pandemic survivors
		'''
		hist = []
		for i in self.pandemic_history:
			hist.append(i[-last:])

		return hist

	def get_creatures(self):
		'''
		Returns a list with three items (healthy, contaminated, dead), which are lists of two lists with x- and y-coordinates of every creature stacked.
		Pandemic use only.
		'''
		creature_list = [[[],[]],[[],[]],[[],[]]] # 0: Healthy, 1: Contaminated, 2: Dead
		for creature in (self.creatures):
			if creature.is_dead:
				creature_list[2][0].append(creature.position[0])
				creature_list[2][1].append(creature.position[1])
			elif creature.is_contaminated:
				creature_list[1][0].append(creature.position[0])
				creature_list[1][1].append(creature.position[1])
			else:
				creature_list[0][0].append(creature.position[0])
				creature_list[0][1].append(creature.position[1])

		return creature_list

	def get_hotspot(self):
		'''
		Returns a list with x- ,y-coordinate if a hotspot exists. If not the case False will be returned.
		'''
		return(self.creatures[0].hotspot)


		return(self.creatures[0].hotspot_sigma)
	def get_plots(self, last=0):
		'''
		Returns a list of the population amounts of all creatures and their color since beginning of the simulation or a specific number of last steps. 
		
		Format: [["color":[population]]]
		'''

		return [[i[0],i[1][-last:]] for i in self.plots]

	def get_x_axis(self, last=0):
		'''
		Returns the x-axis a list since beginning of the simulation or a specific number of last steps. This function is used to plot the population.
		'''
		
		x = []
		if self.pandemic:
			x = [i for i in range(len(self.pandemic_history[0]))]
		else:
			x = [i for i in range(len(self.plots[0][1]))]

		return x[-last:]

	def set_observers(self, observers):
		self.observers = observers

	def __str__(self):
		string = ""
		for creature in creatures:
				string += creature.__str__()+"\n"
		
		return string

class MatplotlibView:
	def __init__(self, env, last=0, auto_generated=0) -> None:
		self.env = env
		self.last = last
		self.auto_generated = auto_generated
		plt.style.use('dark_background')
		plt.ion()

	def notify(self):
		# Get the x-axis to plot and get all y_n
		x = self.env.get_x_axis(self.last)
		y_n = self.env.get_plots(self.last)

		plt.title("Generation: " + str(len(self.env.plots[0][1])))
		plt.ylabel("Amount of creatures")
		plt.xlabel("Amount of timesteps")

		# Plot all y_n with their according color
		for y in y_n:
			# print(y[0],y[1][-1],x[-1])
			plt.plot(x, y[1], color=y[0], linewidth=1)
		labels = []
		
		plt.legend(labels, bbox_to_anchor=(1.05, 1), loc='upper left',mode = "expand")
		plt.pause(0.0001)
		if not plt.fignum_exists(0):
			# Figure closed
			exit() # Close program when exit button is closed
		
		# Clear graph
		plt.cla()
		
		# Draw graph
		plt.draw()

class MatplotlibViewPandemic:
	def __init__(self, env, last=0, auto_generated=0) -> None:
		self.env = env
		self.last = last
		self.auto_generated = auto_generated
		plt.style.use('dark_background')
		plt.ion()

	def notify(self):
		# Create figure with size
		plt.figure(num=0)

		# Subplot 1 - Creature position graph
		creature_list = self.env.get_creatures()
		hotspot = self.env.get_hotspot()
		colors = ["blue","green","red"] # 0: Healthy, 1: Contaminated, 2: Dead
		plt.subplot(211)
		plt.cla()
		plt.title("Generation: " + str(len(self.env.plots[0][1])))

		# Set constant side length
		if self.env.boundaries:
			plt.xlim([0,self.env.size])
			plt.ylim([0,self.env.size])

		# Remove numbering
		plt.xticks([])
		plt.yticks([])

		# Add creatures
		for i, v in enumerate(colors):
			if v == "red":
				plt.scatter(creature_list[i][0], creature_list[i][1], color=v, marker='$\u271D$')
			else:
				plt.scatter(creature_list[i][0], creature_list[i][1], color=v, s = 0.6)
				# Add hotspot
		if hotspot != False:
			plt.scatter(hotspot[0], hotspot[1], color= 'white', s = 10)

		# Subplot 2 - Population size graph:
		plt.subplot(212)

		# Get the x-axis to plot and get all y_n
		x = self.env.get_x_axis(self.last)
		y_n = self.env.get_pandemic_plots(self.last)

		plt.ylabel("Amount of creatures")
		plt.xlabel("Amount of timesteps")

		# Plot all y_n with their according color
		for i, y in enumerate(y_n):
			# print(y[0],y[1][-1],x[-1])
			plt.plot(x, y, color=colors[i], linewidth=1)
		labels = ["Healthy", "Contaminated", "Dead"]
		
		plt.legend(labels, loc='upper left')
		plt.pause(0.0001)
		if not plt.fignum_exists(0):
			# Figure closed
			exit() # Close program when exit button is closed
		
		# Clear graph
		plt.cla()
		
		# Draw graph
		plt.draw()

## USER DEFINED SECTION START ##
# You can modify everything in this section to your liking and experimental will.

# Do you want to simulate a pandemic or a population growth
pandemic = True

# Specify all creatures here
#blue_creature = Creature(color="blue",birth_rate=0,death_rate=0.5,reproduction_rate=0.5, mutations_dict={0:0.5}, start_amount=100)
#red_creature = Creature(color="red",birth_rate=0,death_rate=0.1,reproduction_rate=0.1, mutations_dict={2:0.1})
#green_creature = Creature(color="green",birth_rate=0,death_rate=0.1,reproduction_rate=0.1)


# List all creatures to include here
creatures = []

# Specify how many creatures should be auto-generated
auto_generated = 1

# Set starting population size of auto generator
starting_population = 10

# Specify how many timesteps into the past you should be able to see (0 = disabled)
last = 0

# Specify if you want to precalculate a certain amount of steps (0 = disabled)
precalculate = 0

# Specify the amount of steps the simulation should take
steps = 5000

# Specify field size
size = 10

# Creatures are not able to exit the defined space if set to true
boundaries = True

# Specify movement mode [0: teleport, 1: linear random movement, 2: linear random movement and hotspots]
movement_mode = 2

## USER DEFINED SECTION END ##

def controller(steps, last, precalculate, creatures, auto_generated, size, movement_mode):
	# Auto generate
	mp = MultiplePopulations(auto_generated, pandemic=pandemic, starting_population=starting_population)

	for i in range(len(mp.multiple_population)):
		creatures.append(mp.multiple_population[i])

	# Define the environment
	env = Environment(creatures, pandemic=pandemic, size=size, movement_mode=movement_mode, boundaries=boundaries)

	observers = [MatplotlibViewPandemic(env, last, auto_generated)]
	env.set_observers(observers)

	# Calculate as many steps as defined before starting the animation
	for i in range(precalculate):
		env.step()

	for i in range(steps):
		# Perform a timestep
		env.step()

if __name__ == "__main__":
	controller(steps, last, precalculate, creatures, auto_generated, size, movement_mode)
