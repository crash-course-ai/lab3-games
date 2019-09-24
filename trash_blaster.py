import pygame, math, time, random, copy, multiprocessing, sys, json
import numpy as np
from pygame.locals import *


class PlayerBlaster:
    def __init__(self, game, position: pygame.Vector2, moveDirection: pygame.Vector2):
        """
        Create a new blast object inside the game, with position and direction
        """
        # Download the blaster image
        self.blaster_img = pygame.image.load('./assets/blast.png')

        self.game = game
        self.position = position
        self.rotation = 0.0
        self.radius = 2.0
        self.scale = 0.5
        self.drawTexture = pygame.transform.rotozoom(self.blaster_img, self.rotation, self.scale)
        self.moveDirection = moveDirection
        self.moveSpeed = 500
        self.passedTime = 0.0
        self.lifeTime = 0.5
        # Add this new object to the list of blaster objects in the game.
        self.game.add_player_blaster(self)

    def update(self, dt):
        """
        Every time the game updates, the blast needs to move from its current
        position to a new position according to its speed and direction.
        """
        if self.moveDirection.length_squared() != 0:
            self.moveDirection.scale_to_length(self.moveSpeed)
        self.position += self.moveDirection * dt

        # the wrap_coords function makes the game "toroidal", which means that the
        # blaster will wrap around from one side of the game to the other.
        TrashBlaster.wrap_coords(self.position)
        self.passedTime += dt

        # If the bullet has not hit anything after some time, remove it.
        if self.passedTime >= self.lifeTime: self.destroy()

    def render(self, surface: pygame.Surface):
        """
        Draw the blast image at the new location
        """
        self.drawTexture = pygame.transform.rotozoom(self.blaster_img, self.rotation, self.scale)
        surface.blit(self.drawTexture, self.position - pygame.Vector2(self.drawTexture.get_rect().size) / 2.0)

    def destroy(self):
        """
        Remove the blast from the game
        """
        self.game.remove_player_blaster(self)

    def get_hit(self):
        """
        Action to perform when blast hits something.
        """
        self.destroy()


class JohnGreenBot:
    def __init__(self, game):
        """
        Create a new JohnGreenBot object, add it to the game.
        """
        self.john_green_bot_img = pygame.image.load('./assets/john_green_bot.png')

        self.game = game
        self.position = pygame.Vector2(400, 400)
        self.rotation = 45.0
        self.radius = 18.0
        self.scale = 0.5
        self.drawTexture = pygame.transform.rotozoom(self.john_green_bot_img, self.rotation, self.scale)
        self.moveDirection = pygame.Vector2(0, 0)
        self.moveSpeed = 100
        self.is_blasting = False
        self.shootAccum = 0.0
        self.shootPeriod = 0.25

    def update(self, dt):
        """
        Move the object. Check if blasting. This function is called at every game
        state
        """
        if (self.moveDirection.length_squared() != 0):
            self.moveDirection.scale_to_length(self.moveSpeed)
        self.position += self.moveDirection * dt
        TrashBlaster.wrap_coords(self.position)
        self.shootAccum += dt
        self.blast()

    def blast(self):
        """
        Check if blasting. Create a new blaster object if so.
        """

        # If not blasting
        if not self.is_blasting or self.shootAccum < self.shootPeriod:
            return False

        # If blasting
        self.shootAccum = 0
        shootDirection = pygame.Vector2(-math.cos(math.radians(self.rotation)), math.sin(math.radians(self.rotation)))
        blaster = PlayerBlaster(self.game, self.position + shootDirection * self.radius, shootDirection)

        # Record blast in the game state
        self.game.blasts += 1
        return True

    def render(self, surface: pygame.Surface):
        """
        Draw John Green Bot at the new location
        """
        self.drawTexture = pygame.transform.rotozoom(self.john_green_bot_img, self.rotation, self.scale)
        surface.blit(self.drawTexture, self.position - pygame.Vector2(self.drawTexture.get_rect().size) / 2.0)

    def get_hit(self):
        """
        Perform action when John Green Bot is hit
        """
        self.game.lose()


class Background:
    def __init__(self, game):
        """
        Create the background object
        """
        self.game = game
        self.background_img = pygame.image.load('./assets/background.png')

    def render(self, surface: pygame.Surface):
        """
        Render the background
        """
        surface.blit(self.background_img, pygame.Vector2(0, 0))


class Scoreboard:
    def __init__(self, game):
        """
        Create the scoreboard object
        """
        self.game = game

    def render(self, surface: pygame.Surface):
        """
        Print the score at the top of the game.
        """
        text = str(round(self.game.score, 2)).rjust(6)
        displayFont = pygame.font.Font(pygame.font.match_font("Consolas,Lucida Console,Mono,Monospace,Sans"), 20)
        textImage = displayFont.render(text, True, (255, 255, 255))
        surface.blit(textImage, pygame.Vector2(0, 0))


class Trash:
  def __init__(self, game, position:pygame.Vector2, moveDirection:pygame.Vector2, size):
    """
    Create trash object. Add it to the game with position and direction Give
    it random speed.
    """
    self.trash_img = pygame.image.load('./assets/trash.png')
    self.game = game
    self.size = size
    self.position = position
    self.rotation = 0.0
    self.radius = 18.0 * 1.3**size
    self.scale = 0.20 * 1.3**size
    self.drawTexture = pygame.transform.rotozoom(self.trash_img, self.rotation, self.scale)
    self.moveDirection = moveDirection
    self.moveSpeed = 150 * 0.9**size * random.uniform(0.75, 1.0)
    self.game.add_trash(self)

  def update(self, dt):
    """
    Move the trash
    """
    if self.moveDirection.length_squared() != 0:
        self.moveDirection.scale_to_length(self.moveSpeed)
    self.position += self.moveDirection * dt

    # Make sure that trash behaves toroidally.
    TrashBlaster.wrap_coords(self.position)

  def render(self, surface:pygame.Surface):
    """
    Draw the trash
    """
    self.drawTexture = pygame.transform.rotozoom(self.trash_img, self.rotation, self.scale)
    surface.blit(self.drawTexture, self.position - pygame.Vector2(self.drawTexture.get_rect().size) / 2.0)

  def destroy(self):
    """
    Remove the trash
    """
    self.game.remove_trash(self)

  def split(self):
    """
    Split the trash into two pieces
    """
    rotateAmount = random.uniform(5, 15)

    # Create two new pieces of trash with slightly smaller size
    Trash(self.game, copy.copy(self.position), self.moveDirection.rotate(rotateAmount), self.size - 1)
    Trash(self.game, copy.copy(self.position), self.moveDirection.rotate(-rotateAmount), self.size - 1)

    # Remove the original trash
    self.destroy()

  def get_hit(self):
    """
    What to do if the trash gets hit by the blaster
    """
    self.game.hits += 1
    if self.size >= 2:
        self.split()
    self.destroy()


# Offset for calculating items relative to John Green Bot
OFFSETS = [
    pygame.Vector2(x, y) for x in [-800, 0, 800] for y in [-800, 0, 800]
]


class Specimen:
    def __init__(self):
        """
        Create a specimen (i.e., one of John Green Bot's brains)

        25 inputs: 5 attributes (x, y, x_vel, y_vel, radius) of nearest trash
        objects

        5 outputs: 5 moves (x, y, aim_x, aim_y, blast)
        """
        self.NINPUTS = 25
        self.NOUTPUTS = 5
        self.NINTER = 1
        self.INTERSIZE = 15

        self.inputLayer = np.zeros((self.NINPUTS, self.INTERSIZE))
        self.interLayers = np.zeros((self.INTERSIZE, self.INTERSIZE, self.NINTER))
        self.outputLayer = np.zeros((self.INTERSIZE, self.NOUTPUTS))

        self.inputBias = np.zeros((self.INTERSIZE))
        self.interBiases = np.zeros((self.INTERSIZE, self.NINTER))
        self.outputBias = np.zeros((self.NOUTPUTS))

        self.inputValues = np.zeros((self.NINPUTS))
        self.outputValues = np.zeros((self.NOUTPUTS))


    def save(self, filename):
        fs = open(filename, "w")
        json.dump({
            "inputLayer": self.inputLayer.tolist(),
            "interLayers": self.interLayers.tolist(),
            "outputLayer": self.outputLayer.tolist(),
            "inputBias": self.inputBias.tolist(),
            "interBiases": self.interBiases.tolist(),
            "outputBias": self.outputBias.tolist()
        }, fs)
        fs.close()

    def load(self, filename):
        fs = open(filename, "r")
        data = json.load(fs)
        self.inputLayer = np.array(data["inputLayer"])
        self.interLayers = np.array(data["interLayers"])
        self.outputLayer = np.array(data["outputLayer"])
        self.inputBias = np.array(data["inputBias"])
        self.interBiases = np.array(data["interBiases"])
        self.outputBias = np.array(data["outputBias"])
        fs.close()

    def activation(self, value):
        """
        Activation function, i.e., when to shoot or move.
        """
        return 0 if value < 0 else value

    def evaluate(self):
        """
        Calculate the final output values by evaluating the parameters of the
        speciment. Pass output values through activation function.
        """
        terms = np.dot(self.inputValues, self.inputLayer) + self.inputBias
        for i in range(self.NINTER):
            terms = np.array([self.activation(np.dot(terms, self.interLayers[j, :, i])) for j in
                              range(self.INTERSIZE)]) + self.interBiases[:, i]
        self.outputValues = np.dot(terms, self.outputLayer) + self.outputBias

    def mutate(self):
        """
        Mutate the parameters of the specimen with a probability of 0.05 using a
        Gaussian function with standard deviation of 1. The gaussian function is
        important because it allows most mutations to be small, but a few to be
        very large.
        """
        RATE = 1.0
        PROB = 0.05

        for i in range(self.NINPUTS):
            for j in range(self.INTERSIZE):
                if (random.random() < PROB):
                    self.inputLayer[i, j] += random.gauss(0.0, RATE)
        for i in range(self.INTERSIZE):
            for j in range(self.INTERSIZE):
                for k in range(self.NINTER):
                    if (random.random() < PROB):
                        self.interLayers[i, j, k] += random.gauss(0.0, RATE)
        for i in range(self.INTERSIZE):
            for j in range(self.NOUTPUTS):
                if (random.random() < PROB):
                    self.outputLayer[i, j] += random.gauss(0.0, RATE)

        for i in range(self.INTERSIZE):
            if (random.random() < PROB):
                self.inputBias[i] += random.gauss(0.0, RATE)
        for i in range(self.INTERSIZE):
            for j in range(self.NINTER):
                if (random.random() < PROB):
                    self.interBiases[i, j] += random.gauss(0.0, RATE)
        for i in range(self.NOUTPUTS):
            if (random.random() < PROB):
                self.outputBias[i] += random.gauss(0.0, RATE)

    def calc_fitness(self, doRender=False):
        """
        This function calculates the fitness (i.e., the smartness) of the specimen
        by playing the game and returning the final score.
        """
        game = TrashBlaster()
        return game.run(specimen=self, doRender=doRender)

    def min_offset(self, point1, point2):
        """
        Helper function for apply_input
        """
        candidates = (point2 - point1 + v for v in OFFSETS)
        return min(candidates, key=lambda v: v.length_squared())

    def apply_input(self, game):
        """
        This function takes the game state, loads it into the neural network,
        computes the output, and performs the output actions.
        """
        john_green_bot = game.john_green_bot

        offsets = {a: self.min_offset(john_green_bot.position, a.position) for a in game.trash_list}

        trash_list = sorted(game.trash_list, key=lambda a: offsets[a].length_squared())
        visible_trash = []
        if len(trash_list) > 5: visible_trash = trash_list[0:4]

        # Get all the trash and add them as inputs to the neural network
        for i in range(len(visible_trash)):
            self.inputValues[5 * i + 0] = offsets[visible_trash[i]].x
            self.inputValues[5 * i + 1] = offsets[visible_trash[i]].y
            self.inputValues[5 * i + 2] = visible_trash[i].moveDirection.x if abs(
                visible_trash[i].moveDirection.x) > 0.5 else 0
            self.inputValues[5 * i + 3] = visible_trash[i].moveDirection.y if abs(
                visible_trash[i].moveDirection.y) > 0.5 else 0
            self.inputValues[5 * i + 4] = visible_trash[i].radius

        for i in range(len(visible_trash) * 5, 5 * 5):
            self.inputValues[i] = 0.0

        # Compute the output
        self.evaluate()

        # Actually do the recommended actions
        john_green_bot.moveDirection.x = self.outputValues[0]
        john_green_bot.moveDirection.y = self.outputValues[1]
        john_green_bot.rotation = -math.degrees(math.atan2(self.outputValues[3], self.outputValues[2]))
        john_green_bot.is_blasting = self.outputValues[4] > 0.5


def get_user_rotation(ship):
    toMouse = pygame.mouse.get_pos() - ship.position
    angle = math.atan2(toMouse.y, toMouse.x)
    return -math.degrees(angle)


def get_user_move_direction(ship):
    direction = pygame.Vector2(0, 0)

    keys = pygame.key.get_pressed()

    if (keys[K_w]): direction.y -= 1
    if (keys[K_s]): direction.y += 1
    if (keys[K_a]): direction.x -= 1
    if (keys[K_d]): direction.x += 1

    return direction


def get_user_shoot(ship):
    if pygame.key.get_pressed()[K_SPACE] != 0:
        return True
    return False


def wrap_coords(point):
    dim = (800, 800)
    while point.x < 0: point.x += dim[0]
    while point.x > dim[0]: point.x -= dim[0]
    while point.y < 0: point.y += dim[1]
    while point.y > dim[1]: point.y -= dim[1]
    return point


class TrashBlaster:
    def __init__(self):
        """
        Create new TrashBlaster game with JohnGreenBot, Scoreboard,
        PlayerBlasters, and Trash.
        """
        self.john_green_bot = JohnGreenBot(self)
        self.scoreboard = Scoreboard(self)
        self.background = Background(self)
        self.playerBlasters = []
        self.trash_list = []
        self.score = 0
        self.specimen = None
        self.doRender = True
        self.isPlaying = True
        self.playTime = 0.0
        self.tAccum = 0.0
        self.spawnAccum = 0.0
        self.blasts = 0
        self.hits = 0

    @staticmethod
    def get_user_rotation(john_green_bot):
        toMouse = pygame.mouse.get_pos() - john_green_bot.position
        angle = math.atan2(toMouse.y, toMouse.x)
        return -math.degrees(angle)

    @staticmethod
    def wrap_coords(point):
        """
        Make the game toroidal. That is, if an object touches one side of the
        800 by 800 pixel game board, then move it to the opposite end.
        """
        dim = (800, 800)
        while point.x < 0: point.x += dim[0]
        while point.x > dim[0]: point.x -= dim[0]
        while point.y < 0: point.y += dim[1]
        while point.y > dim[1]: point.y -= dim[1]
        return point

    def run(self, specimen=None, doRender=True):
        """
        Main game loop. Allow the specimen (one of the trained neural networks)
        play the game. Draw and save the game if doRender is True
        """
        self.specimen = specimen
        self.doRender = doRender

        # Create a game surface if we chose to render
        if doRender:
            renderSurface = pygame.display.get_surface()
        else:
            renderSurface = None

            # Record starting time
        t1 = time.time()

        # Make new trash
        self.create_trash()

        # Record number of game frames as count
        count = 0

        # Main loop
        while (self.isPlaying):
            # Update the time
            t2 = time.time()
            # dt is the delta-time, the change in time between game states. The
            # purpose of this variable is to allow the game to play as fast as
            # as the Colab server can handle.
            dt = t2 - t1
            t1 = t2

            if self.specimen and not doRender:
                dt = 0.05

            if doRender:
                self.check_events()

            if not self.specimen or count % 4 == 0:
                # Ask the John Green Bot-specimen for input
                self.apply_input()

            # Update the game state
            self.update(dt)
            self.check_collisions()

            # Make new trash, if needed
            self.create_trash()

            # If we are asking this particular game to be rendered, then take a
            # snapshot of the gameboard and save it.
            if self.doRender:
                self.render(renderSurface)
                pygame.display.flip()
                pygame.display.get_surface().fill((0, 0, 0))
            count += 1

        return self.score

    def check_events(self):
        event = pygame.event.poll()
        while(event.type != NOEVENT):
            if(event.type == QUIT):
                exit()
            event = pygame.event.poll()

    def apply_input(self):
        """
        Ask the specimen (i.e., John Green Bot's neural network) for an action
        """
        if self.specimen:
            self.specimen.apply_input(self)
        else:
            self.john_green_bot.moveDirection = get_user_move_direction(self.john_green_bot)
            self.john_green_bot.rotation = get_user_rotation(self.john_green_bot)
            self.john_green_bot.is_blasting = get_user_shoot(self.john_green_bot)


    def update(self, dt):
        """
        Move the objects in the game by calling their update functions.
        """
        self.john_green_bot.update(dt)
        for playerBlasters in self.playerBlasters:
            playerBlasters.update(dt)
        for trash in self.trash_list:
            trash.update(dt)
        self.playTime += dt
        self.score = self.calc_score()

    def calc_score(self):
        """
        This calculates the score for the game
        """
        # CHANGEME - change the values of this function and watch how John Green
        # Bot's learned behavior changes in response.
        return self.playTime * 1 + self.hits * 10 + self.blasts * -2

    def check_collisions(self):
        """
        Check if a blaster hit some trash, or if some trash hit John Green Bot.
        """
        toHit = set()
        for trash in self.trash_list:
            for playerBlaster in self.playerBlasters:
                if trash.position.distance_squared_to(playerBlaster.position) <= (
                        trash.radius + playerBlaster.radius) ** 2:
                    # A hit occurs if one object is within it's radius' distance to the
                    # other object's radius.
                    toHit.add(trash)
                    toHit.add(playerBlaster)
            if trash.position.distance_squared_to(self.john_green_bot.position) <= (
                    trash.radius + self.john_green_bot.radius) ** 2:
                toHit.add(self.john_green_bot)

        # Call the get_hit function for all objects that are found to have been hit
        for thing in toHit:
            thing.get_hit()

    def create_trash(self):
        """
        Create trash with different size as needed.
        """

        # Adjust diffiulty (number of trash pieces) as time progresses
        if self.playTime <= 20:
            difficulty = 11
        elif self.playTime <= 40:
            difficulty = 14
        elif self.playTime <= 60:
            difficulty = 16
        elif self.playTime <= 80:
            difficulty = 18
        else:
            difficulty = 20

        while len(self.trash_list) < difficulty:
            size = random.randint(1, 4)
            self.spawn_trash(size)

    def spawn_trash(self, size):
        """
        Draw and place the trash objects randomly on the board
        """
        position = pygame.Vector2(0, 0)
        if bool(random.randint(0, 1)):
            if bool(random.randint(0, 1)):
                position = pygame.Vector2(random.uniform(0, 800), 0)
            else:
                position = pygame.Vector2(random.uniform(0, 800), 800)
        else:
            if bool(random.randint(0, 1)):
                position = pygame.Vector2(0, random.uniform(0, 800))
            else:
                position = pygame.Vector2(800, random.uniform(0, 800))

        # Give Trash random direction
        moveDirection = pygame.Vector2(1, 0).rotate(random.uniform(0, 360))
        Trash(self, position, moveDirection, size)

    def render(self, surface: pygame.Surface):
        """
        Draw all the objects in the game
        """
        self.background.render(surface)
        self.john_green_bot.render(surface)
        for playerBlaster in self.playerBlasters:
            playerBlaster.render(surface)
        for trash in self.trash_list:
            trash.render(surface)
        self.scoreboard.render(surface)

    def add_player_blaster(self, blaster):
        self.playerBlasters.append(blaster)

    def remove_player_blaster(self, blaster):
        if (blaster in self.playerBlasters):
            self.playerBlasters.remove(blaster)

    def add_trash(self, trash):
        self.trash_list.append(trash)

    def remove_trash(self, trash):
        if trash in self.trash_list:
            self.trash_list.remove(trash)

    def lose(self):
        """
        If we lose the game, then set the main loop condition to false.
        """
        self.isPlaying = False


def get_fitness(specimen):
    return specimen.calc_fitness()


if __name__ == "__main__":
    pygame.init()

    args = sys.argv[1:]

    gen_size = 200
    saveBest = False
    saveFile = "brain.txt"
    loadFile = ""
    doDisplay = True
    displayMod = 1
    nThreads = 4

    mode = ""

    while len(args):
        arg = args.pop(0)
        if arg == "--display-every":
            arg = args.pop(0)
            displayMod = int(arg)
        elif arg == "--no-display":
            doDisplay = False
        elif arg == "--save-best":
            saveBest = True
        elif arg == "--gen-size":
            arg = args.pop(0)
            genSize = int(arg)
        elif arg == "--save-file":
            arg = args.pop(0)
            saveFile = arg
        elif arg == "--load-file":
            arg = args.pop(0)
            loadFile = arg
        elif arg == "--num-threads":
            arg = args.pop(0)
            nThreads = int(arg)
        elif mode == "":
            mode = arg
        else:
            exit(-1)

    if mode == "":
        mode = 'play'

    if doDisplay:
        pygame.display.set_mode((800, 800))

    if mode == "learn":
        # Start up a bunch of processing threads to do lots of work at the same time.
        pool = multiprocessing.Pool(nThreads)

        # Create gen_size number of NEW  specimen
        generation = [Specimen() for i in range(gen_size)]

        if loadFile != "":
            for specimen in generation:
                specimen.load(loadFile)

        # The map function applies each specimen in the generation to the git_fitness
        # function. Simply put, this next line of code plays 200 games at the same time,
        # and collects their scores.
        scores = pool.map(get_fitness, generation)

        # Create a map of a specimen to its score.
        specimen_score_map = {}
        for i in range(len(generation)):
            specimen_score_map[generation[i]] = scores[i]

        # Sort the specimen by their score and keep only the top half.
        half_size = gen_size // 2
        generation = sorted(specimen_score_map, key=lambda k: specimen_score_map[k], reverse=True)[0:half_size - 1]

        # Initialize an interation variable because the next few cells may be run many times.
        iteration = 0

        try:
            while True:

                # Increment the iteration counter
                iteration += 1

                # For each reproducer, create a copy, mutate it, and add it to the generation
                for i in range(gen_size // 2):
                    child = copy.deepcopy(generation[i])
                    child.mutate()
                    generation.append(child)

                if doDisplay and iteration % displayMod == 0:
                    # Find the top example. Call the calc_fitness function with doRender = True so
                    # that it fills in the images variable with image-captures of the game play
                    example_score = generation[0].calc_fitness(doRender=True)
                else:
                    example_score = 'UNK'

                if saveBest:
                    generation[0].save(saveFile)

                # At this point we have a new generation. Half of these are parents/reproducers
                # and half are mutant-children.

                # The pool.map function calls the get_fitness function defined in STEP 3.1
                # "simultaneously" for each specimen in the generation
                scores = pool.map(get_fitness, generation)

                # Create a map of specimen to its score
                specimen_score_map = {}
                for i in range(len(generation)):
                    specimen_score_map[generation[i]] = scores[i]

                # Find the mean-average of the scores of all specimen
                average_of_all = sum(specimen_score_map.values()) / gen_size

                # Sort the specimen by their score and keep only the top half.
                generation = sorted(specimen_score_map, key=lambda k: specimen_score_map[k], reverse=True)[
                             0:half_size - 1]

                # Find the mean-average of the scores of all reproducers (i.e., the top half of
                # all specimen)
                average_of_reproducers = sum(sorted(specimen_score_map.values(), reverse=True)[0:half_size]) / half_size

                # Print the statistics
                print('ITERATION {}'.format(iteration))
                print('\tAverage score of all specimen: {}'.format(average_of_all))
                print('\tAverage score of reproducers: {}'.format(average_of_reproducers))
                print('\tScore of the video-specimen: {}'.format(example_score))

        except KeyboardInterrupt:
            exit()


    elif mode == "playback":
        specimen = Specimen()
        specimen.load(loadFile)
        while True:
            print("Score: ", specimen.calc_fitness(doRender=True))

    elif mode == "play":
        game = TrashBlaster()
        print("Score: ", game.run())

    elif mode == "help" or mode == "-h" or mode == "--help":
        print("SpaceQ.py MODE [flags]")
        print("\tMODE = learn, play, playback")
        print("\tflags:")
        print("\t\t--no-display")
        print("\t\t--display-every N")
        print("\t\t--save-best")
        print("\t\t--gen-size N")
        print("\t\t--save-file FILE")
        print("\t\t--load-file FILE")
        print("\t\t--num-threads N")

    else:
        print("invalid mode:", mode)
