# Supply_Chain

This is a group project Supply_Chain by Nicole, Paul and Janine :D

## Helpful Git Commands 

### git clone [repository-URL]
Clones the repository to your local machine.
(git clone https://github.com/Cyani180/Supply_Chain.git)

### git branch -a
Shows us all currently available branches known to git in our repository.
We might need to call 'git pull' to get the info of new branches.

### git branch [branch name]
Creates a new branch with a given name.

### git checkout [branch name]
With this command we checkout a branch.

### git status
This is maybe the most important git command. Whenever we don't know in what state we currently are or whenever something is odd, we can just type 'git status' and it will usually display some information that will tell us what is going on.

### git pull
This is used to get changes from the remote repository.
So whenever someone else did something on our branch we can get the changes they did by using this command.
*Sidefact:* This also checks for new branches available on the remote repository and adds them to a list of known branches.

### git add -[Path to a new file] / git add -A
This can be used to add a new existing file or add all new existing files in our repository.
If do not do this new files are not recognized by git and won't be commited.

### git commit -a -m "Add new feature"
This is used to create a commit with all the changes we have done so far.
Files that have not yet been added via 'git add' won't be commited here.
The '-a' declares that we want to add all changes.
The '-m "Some text"' is used to create a commit message.
*Hint:* A commit is not yet remotely available for others until we push those changes

### git push
With this command we push our commits to the remote branch, so that everyone can pull them and get the new changes.
When doing this for the first time it should usually fail.
We first need to call 'git push -u origin [current branch name]': This will create a remote branch that tracks our local branch.


## More advanced Git Commands

### git rebase -i main
This basically takes the main branch and puts all the changes (all the commits) of our current branch behind those.
This gets us new commits from the main branch that were not existing when we created our current branch and is used to ensure that our commit-history stays in order on our main branch.
When later merging the current branch to the master branch it also ensures that we get less to maybe no merge conflicts.
When using Merge-Requests on GitHub they will usually do this for us.

### git merge [branch name]
This merges the given branch to the current branch.
We should however propably use Merge-Requests on GitHub that will do this for us.

## Best practise:
Usually our workflow is as follows:
1. 'git pull' on our main branch to get all the new changes and be up-to-date.
2. 'git branch my-new-branch' to create a new branch.
3. 'git checkout my-new-branch' to switch to the new branch.
4. Do something on your branch.
5. Maybe add files with 'git add -A'.
6. 'git commit -a -m "I did something"' to create one or more commits.
7. 'git push -u origin my-new-branch' for the first time or simply 'git push' once we have done 'git push -u ..' once
8. Create a Merge-Request on GitHub
9. Someone merges the changes GitHub
10. 'git checkout main' to get back to the main branch and then we can start over from point 1 for the next thing we want to add.
