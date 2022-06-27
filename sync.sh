#!/bin/bash

HOST='staff.ustc.edu.cn'
USER='zqj'
PASS='ok2if2An'
LCD='/home/zqj/work_dir/qijing_git/jekyll-theme-chirpy/_site'
RCD='public_html'

# user $USER $PASS
lftp -f "
open -u ${USER},${PASS} $HOST
lcd ${LCD}
mirror --delete --reverse --verbose ${LCD} ${RCD} 
bye
"
