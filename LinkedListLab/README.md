# clab
This is the handout directory for [clab](https://github.itu.dk/pages/OSC/23-Website/labs/clab.pdf) - a lab that drills you in the C style of programming. Make sure to read the above-URL'ed PDF before you proceed.

This README details how you should handle this assignment:

1. [Setting up clab](#setting-up-clab)
2. [Doing the assignment](#doing-the-assignment)
3. [Submitting your work](#submitting-your-work)


## Setting up clab
You must fork this repository to your own user account and work on the
assignment from that fork. When you are done, you should create a pull request
against the original repository.


### Fork repository
The following images illustrate how to fork the course repository.

First press the Fork-button on the lab repository:
![Fork](https://github.itu.dk/pages/OSC/23-Website/images/Fork.png)


You may have to select the organisation or owner of the fork.<br>
Select your ITU username:
![ChooseUser](https://github.itu.dk/pages/OSC/23-Website/images/ChooseUser.png)


When forked, we need to get the repository’s URL to clone from.<br>
If you press `Use HTTPS` at the top of the popup box (see image below), you
will authenticate against github with ITU username and password instead of
ssh keys (see [this link](https://devconnected.com/how-to-setup-ssh-keys-on-github/)
for information on how to use SSH keys to interact with GitHub).
![GetCloneUrl](https://github.itu.dk/pages/OSC/23-Website/images/GetCloneUrl.png)


### Setup your working copy of the repo
Switch to a terminal (logged in to the course server, for instance) and do the
following:
```
# Download git repository (replace nhed with your own user name)
git clone git@github.itu.dk:nhed/23-clab.git clab

# Change into cloned directory
cd clab

# Checkout your branch (replace nhed with your own user name)
git checkout -b nhed

# Make initial push to remote to set upstream clone URL (replace nhed with your own user name)
git push -u origin nhed
```

## Doing the assignment
You should now have access to the course files on your user account and have set
up a tracking branch corresponding to your username on your own fork of the
repository.<br>
You are now ready to edit `queue.c` and `queue.h` according to the instructions in the
assignment [pdf file](https://github.itu.dk/pages/OSC/23-Website/labs/clab.pdf) and the handout’s original
[README](./README).

Example workflow:
```
# Open and edit the assignment file in vim
vim queue.c

# Clean out previously built files
make clean

# Compile the `qtest` testing facility
make

# Run the `qtest` testing facility (to manually test individual commands)
make run

# Run the driver (to test your implementation on the provided test suite)
make test

# Commit your work to version control.
git add queue.c queue.h
git commit -m “Solved exercise foo by doing bar.”
```    

It is good practice to do the last step regularly.<br>
Try making a habit out of committing each new thing that you try out -preferably
when it works- to more effectively use your revision history.


## Submitting your work
When you are ready to submit the assignment, you should commit any outstanding
changes to `queue.c` and `queue.h`, and push them to your fork. Then, create a pull request to
the “forked-from” repository, as illustrated below.


### Push changes to your branch
Add changed `queue.c`- and `queue.h`-file to staging area, then commit staged file:
```
git add queue.c queue.h
git commit -m “Hand-in from nhed. Final version.”:
```

Push changes to your fork:
```
git push
```
   
### Create a pull request
On ITU’s github, on your fork’s repository frontpage
(`https://github.itu.dk/<YOUR_ITU_ALIAS>/23-clab`) you can see how
many commits you have made since forking and there is a little button in the
same line that allows you to make pull requests against the repository and
branch that was forked from:

![New PR](https://github.itu.dk/pages/OSC/23-Website/images/MakePullRequest.png)


This will take you to the following page, where you should:

1. Enter your initials in the title.
2. Type in comments to your handins (sorta like assignment comments on LearnIT)
3. Press the green `Create pull request`-button<br>
   (alternatively, you can press the downward arrow right next to it and only
   open the PR in `DRAFT` mode indicating that it is not yet ready for TA
   review).


![Open PR](https://github.itu.dk/pages/OSC/23-Website/images/OpenPullRequest.png)
