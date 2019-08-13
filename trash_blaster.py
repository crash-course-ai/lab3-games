import pygame, math, time, random


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
        800 by 800 pixel game board, then move it to the opposide end.
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
                data = pygame.image.tostring(window, 'RGBA')
                display_img(data)
            count += 1

        return self.score

    def apply_input(self):
        """
        Ask the specimen (i.e., John Green Bot's neural network) for an action
        """
        self.specimen.apply_input(self)

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