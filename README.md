# kaggle-Shopee-Price-Match-Guarantee
Kaggleのshopeeコンペのリポジトリ
<img width="946" alt="スクリーンショット 2021-04-12 22 08 49" src="https://user-images.githubusercontent.com/70050531/114399570-08579500-9bdc-11eb-95e0-2b519858cdc5.png"><br>
Kaggleの Shopee - Price Match Guarantee コンペのリポジトリです。<br>
nbというディレクトリに、今回使用したNotebookをおいてあります。<br>
ただし、下の方針にもある通り、今回はKaggle上でほぼ完結させているため、gitは使用していません。ですので、nbの中に全てのversionのNotebookがあるわけではないです。ver名がないものに関しては最新のverになります。<br>

## 方針
- 基本的にはKaggle上で実験を行い、quotaが切れたらローカルのGPUで実験を行う。
- Version名は常にVersion〇〇に統一して、変更などはKaggle日記(このリポジトリのREADME.md)に書き込む。<br> 

## Paper<br>
- 参考にした論文の一覧。決して全てを理解してるわけではない。<br>

| No | Name | Detail | Date | link |
| ----- | ----- | ----- | ----- | ----- |
| 01 | ArcFace: Additive Angular Margin Loss for Deep Face Recognition | CNNの最後の全結合層の代わりにArcMarginProductを入れることで、特徴良マップの距離を各クラスごとに離すことに成功し、顔認証のSOTAを記録した。 | 9 Feb 2019 | [link](https://arxiv.org/pdf/1801.07698.pdf) | 

## Basics<br>
### Overview(Deepl)<br>
あなたは、お得な情報を求めてオンラインショップをチェックしていますか？同じ商品を買うのに、店によって値段が違うのは嫌だと思う人は多いでしょう。小売企業は、自社の商品が最も安いことをお客様に保証するために、さまざまな方法を用いています。中でも、他の小売店で販売されている商品と同等の価格で商品を提供する「プロダクトマッチング」という手法があります。このマッチングを自動的に行うには、徹底した機械学習のアプローチが必要であり、データサイエンスのスキルが役立ちます。<br>

似たような商品の2つの異なる画像は、同じ商品を表している場合もあれば、全く別の商品を表している場合もあります。小売業者は、2つの異なる商品を混同してしまうことで生じる誤表示などの問題を避けたいと考えています。現在は、ディープラーニングと従来の機械学習を組み合わせて、画像やテキストの情報を解析し、類似性を比較しています。しかし、画像やタイトル、商品説明などに大きな違いがあるため、これらの方法は完全には有効ではありません。<br>

Shopeeは、東南アジアと台湾における主要なEコマースプラットフォームです。お客様は、それぞれの地域に合わせた、簡単で安全かつ迅速なオンラインショッピング体験を高く評価しています。また、Shopeeに掲載されている何千もの商品に対して「最低価格保証」機能を提供するとともに、強力な支払いサポートと物流サポートを提供しています。<br>

このコンペティションでは、機械学習のスキルを応用して、どのアイテムが同じ商品であるかを予測するモデルを構築します。

その応用範囲は、Shopeeや他の小売業者をはるかに超えています。商品のマッチングに貢献することで、より正確な商品分類をサポートし、マーケットプレイスのスパムを発見することができます。顧客は、買い物をする際に、同じ商品や似たような商品がより正確にリストアップされることで利益を得ることができます。そして最も重要なことは、あなたやあなたの周りの人たちがお得な商品を探すのに役立つということです。<br>

### data(DeepL)<br>
大規模なデータセットの中から重複しているものを見つけることは、多くのオンラインビジネスにとって重要な問題です。Shopeeの場合、ユーザーが自分で画像をアップロードしたり、商品説明を書いたりすることができるので、さらに多くの課題があります。あなたの仕事は、どの商品が繰り返し投稿されているかを特定することです。関連する商品の違いは微妙ですが、同じ商品の写真は大きく異なります。<br>

これはコードコンペなので、テストセットの最初の数行/画像のみが公開され、残りの画像は提出されたノートブックでのみ見ることができます。隠されたテストセットには、約 70,000 枚の画像が含まれています。公開されている数行のテストと画像は、隠しテストセットのフォーマットとフォルダ構造を説明するためのものです。<br>

#### Files<br>
[train/test].csv - トレーニングセットのメタデータです。各行には、1つの投稿のデータが含まれています。複数の投稿は、画像IDが全く同じでもタイトルが違ったり、その逆の場合もあります。<br> 

- posting_id - 投稿のIDコードです。<br>
- image - 画像のID/md5sumです。<br>
- image_phash - 画像のパーセプチュアルハッシュです。<br>
- title - 投稿の商品説明です。<br>
- label_group - 同じ商品に対応するすべての投稿のIDコード。テストセットでは提供されていません。<br>

[train/test]images - 投稿に関連する画像。<br>

sample_submission.csv - 正しいフォーマットのサンプル投稿ファイルです。<br>

- posting_id - 投稿物のIDコードです。<br>
- matches - この投稿にマッチするすべての投稿IDのスペース区切りのリストです。投稿は常にセルフマッチします。グループサイズの上限は50なので、50以上のマッチを予測する必要はありません。<br>

## Log<br>
### 20210411<br>
- join!!!<br>
- 今度こそ最後まで走り切るぞ。データセットのサイズも良心的だしきっといけるはず。<br>
- yyamaさんとチームを組む約束をした。yyamaさんが来てくださるまでの間に色々進めておくことにする。<br>
- nb001(EDA)<br>
  - ver1<br>
    - 画像サイズが意外にも大きい。1000x1000を超えてくるやつもある。最後にsizeを上げるのはアリかもしれない。<br>
    - 画像のペアとしては2が圧倒的に多く、指数的に減少していく感じ。多クラス分類として解くとクラス数が10000を超えてしまうのでかなり多いか。<br>
    - imageが全く同じものもあった。それらは別に学習なんかしなくても自動的に見分けられるので、最後にそれは一応チェックしてもいいかもしれない。ただ、testデータが見えるわけではないのであまり意味はなさそう。<br>
    - shopeeは東南アジア、台湾系のメルカリ的な存在だと思われるので、商品の説明(title)は割と汚い。BERTなどのtransformerを使うのであれば綺麗にしなければいけない。<br>
### 20210412<br>
- nb001<br>
  - ver2<br>
    - [公開Notebook](https://www.kaggle.com/ishandutta/v7-shopee-indepth-eda-one-stop-for-all-your-needs)を参考にして追加でEDAを行った。titleの前処理がとても綺麗だったので、textで攻める時には参考にできそう。<br>
    - 各カラムのnuniqueの値を調べた。imageとimage_phashでnuniqueの値が違った(前者の方が大きい)。これはimageは違うけどimage_phashが等しいということがあることを意味する。どういうこと...?<br>
- nb002(training_EfficientNetB3ns)<br>
  - ver1<br>
    - [LBが0.714のこの公開Notebook](https://www.kaggle.com/tanulsingh077/pytorch-metric-learning-pipeline-only-images)と[LBが0.728のこの公開Notebook](https://www.kaggle.com/vatsalmavani/shopee-training-eff-b4)を参考にして、Y.Nakamaさんのパイプライン主体でベースラインを書いた
    - 前者と後者の違いがいまいちよくわからなかった。後者の方がモデルが大きいことと、epochが大きいことに起因しているのか?後は、後者はローカルマシンでbatch_sizeを上げてモデルを作っているっぽい。それも効いているかもしれない。<br>
    - ArcFaceというレイヤーを最後の全結合層の代わりに差し込むことで、各クラスの距離を離すように学習できるらしい。[原論文](https://arxiv.org/pdf/1801.07698.pdf)と[Qiita記事](https://qiita.com/yu4u/items/078054dfb5592cbb80cc)を参考にして理解した。<br>
    - validation_lossが全く下がらなかったので一旦止めた。<br>
  - ver2<br>
    - 原因はfoldの切り方にあると判断したので、foldの切り方を色々調べた。<br>
    - [先程の公開Notebook(前者)](https://www.kaggle.com/tanulsingh077/pytorch-metric-learning-pipeline-only-images)や[Vote数が高いdiscussion](https://www.kaggle.com/c/shopee-product-matching/discussion/228794)を見ると、label_groupでgroupKFoldをするのが支配的っぽい。discussionでは、その上でさらにlabel_groupの数でstratifiedにもしたほうがいいと述べられていた。RANZCRでも最後の最後でPatientIDでstratifiedにした方がいいということに気づいたので、今回はCV戦略は大事にしたい。<br>
    - [このサイト]でStratifiedGroupKFoldについて詳しく書かれていた。実装してみたが、うまく動かせなかったので、とりあえず途中でセーブした。。<br>
  - ver3<br>
    - ひとまずver1に戻して動かした。epochsを10に、max_grad_normを50に変えてある。<br>
    - やはり、train_lossのみが減って、valid_lossは増えてしまった。CVの切り方をもっと調べることにする。<br>

### 20210413<br>
- nb002<br>
  - ver4<br> 
    - もともとの[公開Notebook](https://www.kaggle.com/tanulsingh077/pytorch-metric-learning-pipeline-only-images)では、GroupKFoldが用いられていると思いきやStratifiedKFoldが用いられてた。確かに、今回は11014クラスの多クラス分類で解いているため、全部知らないグループの中では全く予測できないよねと納得した。<br>
    - StratifiedKFloldを用いるとvalid_lossが減少した。foldの切り方も公開されているものと同じっぽいので、これでいいんだと思われる。<br>
    - 最終的には、image_phaseかimageのいずれかをgroupにして、label_groupをstratifiedにするのが良さそうだが、とりあえずは現状のCVとLBのチェックが先。<br>
