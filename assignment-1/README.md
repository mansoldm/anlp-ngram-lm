##### Clone repository
```
git clone https://github.com/mansoldm/anlp-coursework.git
```
##### Making changes
After making changes to one or more files, stage them
```
git add file1 file2 ... filen
```
can also add them one by one or use regex

Create a commit with these changes (don't forget the `-m`).
```
git commit -m "type what you did here"
```
Finally push the commit to the remote
```
git push
```
Pushing won't work if you haven't pulled new commits that I have already pushed
##### Getting new changes
To get a commit that somebody else has pushed
```
git pull
```
Only execute this command when you don't have any outstanding changes. Not too much of an issue, just makes things easier/avoids unnecessary headaches
