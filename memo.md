```
nvubu ~/p/d2vg ❱ http -pb get 'http://127.0.0.1:8000/items/5?q=someq'
{
    "item_id": 5,
    "q": "someq"
}
```

wget --recursive --level 3 --no-clobber --random-wait --restrict-file-names=windows --convert-links --no-parent --adjust-extension https://ja.stackoverflow.com/

途中から「too may requests」というエラーでダウンロードさせてくれなくなった。

python3 d2vg.py "csvファイルを整形する" 'sample-texts/ja-stackoverflow/ja.stackoverflow.com/questions/**/*.html'

python3 d2vg.py "図形を認識する" 'sample-texts/ja-stackoverflow/ja.stackoverflow.com/questions/**/*.html'

cd ja.stackoverflow.com
python3 ../expand_ja_stackoverflow_com_posts.py

