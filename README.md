# lab3-games

## Purpose
**Trash Blaster** is a simple game written in Python meant to be used to explore machine learning techniques. It was created as part of the *Crash Course AI* series on Youtube.

## Contents
The `trash_blaster.py` file contains the game as well as the machine learning code.
The `bigbrain.txt` file contains data from a machine which learned how to play **Trash Blaster** over more than 100,000 iterations of 200 games each. Read below to see how to run it with `trash_blaster.py` on your computer. 

## Before You Start
You must install Python 3 on your computer, as well as the following Python modules:

`numpy`

`pygame`

The modules can be installed by typing the following in your command line:

`python3 -m pip install numpy pygame`

**Depending on how Python is installed on your computer, you may have to replace `python3` with `python` in the command line.**

## How to Use `trash_blaster.py`
From the command line, type

`python3 trash_blaster.py MODE`

but replace `MODE` with one of `play`, `learn`, or `playback`. The functionality of each of these is described below.

### `play` mode
This mode allows you to play **Trash Blaster** yourself, to test your skill. Move using the `WASD` keys, aim with the mouse pointer, and fire the blaster with the spacebar.

There are no flags applicable to this mode.

Example: `python3 trash_blasher.py play`

### `learn` mode
This mode allows you to train an AI to play **Trash Blaster**. 

Use the `--save-best` flag to make the program save the best AI "brain" each iteration.

Use the `--save-file` flag followed by a filename to determine the name of the file `--save-best` uses (`brain.txt` by default).

Use the `--load-file` flag followed by a filename to use the "brain" in that file as the starting point.

Use the `--no-display` flag to run the learning algorithm without opening a game window visible to the user. This is recommended to save time on long training sessions.

Use the `--display-every` flag followed by an integer to determine how many learning iterations pass between games shown to the user.

Use the `--num-threads` flag followed by an integer to determine how many threads should be used in the learning algorithm (`4` by default).

Example: `python3 trash_blaster.py learn --save-best --save-file mybrain.txt --display-every 5 --num-threads 2`

### `playback` mode
This mode allows you to play a previously-trained "brain" on randomly-generated games to see how it performs.

Use the `--load-file` flag followed by a filename to play the "brain" in that file. You must use this flag.

Example: `python3 trash_blaster.py playback --load-file bigbrain.txt`