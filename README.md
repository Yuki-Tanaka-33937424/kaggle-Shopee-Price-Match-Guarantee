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
| 02 | EfficientNetV2: Smaller Models and Faster Training | EfficientNetにFused-MBConv(depthwiseとpointwiseではなく普通の畳み込み)も入れて、スケーリングの仕方を変えつつ、progressive-learning(画像サイズと正則化を両方大きくしていく学習方法)を採用して、学習効率と精度の両方を改善した。 | 1 Apr 2021 | [link](https://arxiv.org/pdf/2104.00298.pdf) | 

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
- EfficientNetV2のweightが追加されたらしい。[原論文](https://arxiv.org/pdf/2104.00298.pdf)と[Qiita記事](https://qiita.com/omiita/items/1d96eae2b15e49235110)を見るとかなり強そうなので、CVの切り方が安定したらモデルをこれに切り替えてみたい。<br>
- ArcFaceより強いっぽい手法についての論文を見つけた([原論文のリンク](https://arxiv.org/pdf/2101.09899v2.pdf))。後で取り入れる。<br>
- nb002<br>
  - ver4<br> 
    - もともとの[公開Notebook](https://www.kaggle.com/tanulsingh077/pytorch-metric-learning-pipeline-only-images)では、GroupKFoldが用いられていると思いきやStratifiedKFoldが用いられてた。確かに、今回は11014クラスの多クラス分類で解いているため、全部知らないグループの中では全く予測できないよねと納得した。<br>
    - StratifiedKFloldを用いるとvalid_lossが減少した。foldの切り方も公開されているものと同じっぽいので、これでいいんだと思われる。<br>
    - 最終的には、image_phaseかimageのいずれかをgroupにして、label_groupをstratifiedにするのが良さそうだが、とりあえずは現状のCVとLBのチェックが先。<br>
    - しっかりLossが落ちた。<br>
    - | train_loss | valid_loss | 
      | :---: | :---: |
      | 0.0652 | 1.8025 | <br> 
    - かなり過学習している様子なので、何かしらの対策が必要か。ただ、valid_lossは最後まで単調減少していた。<br>
- nb003(EfficientNetB3ns_inference_with_Tfidf)<br>
  - ver1<br>
    - [LB0.712の公開Notebook](https://www.kaggle.com/tanulsingh077/metric-learning-image-tfidf-inference?scriptVersionId=58359119)と[LB0.728の公開Notebook](https://www.kaggle.com/vatsalmavani/eff-b4-tfidf-0-728/comments?scriptVersionId=59449258)を参考にして書く。<br>
    - foldごとにCVを計測できるように仕掛けたが、f1スコアが全く上がらない。全体でCVを出したらうまくいくんだけど...<br>
    - 理由がわかった。学習の段階ではlabel_groupを当てにいっていたため、仮にvalidationの中にペアのうちの片方しかなくてもうまく行っていた(というか、stratifiedの対象がlabel_groupだったので故意にそうしていた)。それに対し、CVを計測する段階ではペアを当てなければいけないため、データの中にペアが必ずいなくてはいけない。ということは、foldごとに分けられないということだ。したがって、全体でペアを探した後にvalidationのデータだけを切り取ればいいということか。<br>
    - text_predictionsにはKNNではなくcosine similarityを使っている。<br>
  - ver4(ver2, 3は失敗)<br>
    - 書き終えたが、バッチサイズが大きくて(64)推論の過程でGPUでKNNをやったときにOOMを吐かれたらしい。<br>
  - ver5<br>
    - やっと実行できた。<br>
    - | train_ver | CV | LB |
      | :---: | :---: | :---: | 
      | 4 | 0.7532 | 0.715 | <br> 
    - CVとLBの乖離が大きい。やはりCVの切り方は考えないといけない。<br>

### 20210415<br>
- [BERTを使った公開Notebook](https://www.kaggle.com/zzy990106/b0-bert-cv0-9)と[論文](https://www.aclweb.org/anthology/2020.ecomnlp-1.7.pdf)があった。参考にしたい。<br>
- nb002<br>
  - ver6(ver5は失敗)<br> 
    - batch_sizeを変えても学習率を変えなくて済むように、lr = Constant * batch_sizeの形式に変更した。値としては若干大きくなっている。<br>
    - | train_loss | valid_loss |
      | :---: | :---: |
      | 0.0616 | 1.7975 | <br> 
    - 両方若干よくなった。<br>
    - ちなみに、CosineAnnealingLRは明らかにダメそうだった。最初に過学習してしまうみたい。<br>
  - ver8(ver7は失敗)<br>
    - foldをGroupKFoldにしてみた。学習の段階では多クラス分類として解かざるを得ないためにvalidation_lossは下がらないが、CVは正確になると踏んだ。<br>
    - | train_loss | valid_loss |
      | :---: | :---: |
      | 0.0510 | 12.4228 | <br> 
    - 予想通りvalid_lossは全く落ちてないが今は気にしない。<br>
  - ver10(ver9は失敗)<br>
    - ver6にSAMを加えてbatch_sizeを12に落とした。<br>
    - | train_loss | valid_loss | 
      | :---: | :---: |
      | 0.1289 | 1.9122 | <br>
    - バッチサイズを落とした影響で、学習率も(線形ではあるが)落ちているので、学習が終わりきっていないように見える。恐らく、十分にバッチサイズを大きくできれば結果は変わると思われる。<br>
- nb003<br>
  - ver6<br>
    - text_predictionsでKNNを使えるようにコードを書いたが、失敗してver5と全く同じスコアを出してしまった。<br>
  - ver8(ver7は失敗)<br>
    - やっとできた。<br>
    - | train_ver | CV | LB |
      | :---: | :---: | :---: | 
      | 4 | 0.7546 | - | <br> 
    - 再びエラーになった。やはりバッチサイズは16に下げた方がいいっぽい。<br>
  - ver9<br>
    - [この公開Notebook](https://www.kaggle.com/ragnar123/shopee-inference-efficientnetb1-tfidfvectorizer/comments)を参考に、get_text_embeddingsの中のpcaを外した。<br>
    - 上記の公開Notebookでは、GroupKFoldを使っていた。悩ましい...<br>
    - | train_ver | CV | LB |
      | :---: | :---: | :---: | 
      | 4 | 0.7641 | - | <br> 
    - またエラーになった。今度はtext_embeddingsのmax_featuresが多すぎてメモリがオーバーしてしまうらしい。<br>
    - 今回は21500にしていたが、動作確認が取れた18000にする。<br>
  - ver11(ver10は失敗)<br> 
    - | train_ver | CV | LB |
      | :---: | :---: | :---: | 
      | 4 | 0.7647 | - | <br> 
    - 結局エラーを吐かれた。これは諦める。<br>
  - ver12<br>
    - ver8から、バッチサイズを8に落とした。<br>
    - | train_ver | CV | LB |
      | :---: | :---: | :---: | 
      | 4 | 0.7546 | 0.691 | <br> 
    - 大撃沈したので今後のベースラインはver5になった。<br>
  - ver13<br>
    - CNNのnb002_ver8のものに変えた。<br>
    - ver12が撃沈したのでこっちもボツ。<br>
  - ver14<br>
    - ver5から、EfficientNetをnb002_ver8のものに変えた。<br>
    - | train_ver | CV | LB |
      | :---: | :---: | :---: | 
      | 8 | 0.667 | 0.662 | <br>
    - CV、LBともにかなり下がったが、CVとLBがほぼ同じ値になった。方針的としては当たっているように思える。<br>
### 20210416<br>
- CVの切り方がよくわからない。[このディスカッション](https://www.kaggle.com/c/shopee-product-matching/discussion/229794)や[このディスカッション](https://www.kaggle.com/c/shopee-product-matching/discussion/225664)などでも話されているが、GroupKFoldでCVが0.7~0.8程度出ているのが再現できないのが気になる。恐らくstratifiedよりはgroupの方がいいのだろうが、何が違うのかわからない。<br> 
- nb002<br>
  - ver12<br>
    - ローカルのRTX3090を使った。バッチサイズは24まであげている。<br>
    - | train_loss | valid_loss |
      | :---: | :---: |
      | 0.0420 | 1.7825 | <br>
    - 一応よくなったが思ったほどはよくなってない。<br>
  - ver13<br>
    - せっかくマシンパワーがあるので、epochを15にした。<br>
    - | train_loss | valid_loss | 
      | :---: | :---: | 
      | 0.0280 | 1.7847 | <br>
    - まず過学習をどうにかしないといけない。<br> 
  - ver14<br> 
    - dropout=0.3にした。<br>
    - | train_loss | valid_loss | 
      | :---: | :---: | 
      | 0.0629 | 1.7521 | <br>
  - ver15<br>
    - dropout=0.5にした。<br>
    - | train_loss | valid_loss |
      | :---: | :---: |
      | 0.1124 | 1.7201 | <br>
    - 過学習がいい感じに防がれている。よく確認したらYYamaさんが既にdropout=0.5も試してくれていた。反省。<br>
- nb003<br>
  - ver15<br>
    - nb002_ver11のモデルで作ったが、RTX3090で作り直すことにした。<br>
  - ver16<br>
    - | train_ver | CV | LB |
      | :---: | :---: | :---: | 
      | 12 | 0.756 | 0.716 | <br> 
    - ver5と比べて若干上がってはいるが、誤差と言っていい範囲。<br> 
### 20210417<br>
- nb002<br> 
  - ver16<br>
    - cassavaのNotebookから色々な損失関数を引っ張ってきて動作確認をしてquick saveした。<br>
  - ver17<br>
    - ver15に加えて、GroupKFoldを試した。その時、foldごとにクラス数が変わってしまうため、それに合わせてクラス数も変える必要があった。しかし、それだけだとなぜかcudaのエラーが出てしまっていた。<br>
    - 理由が判明した。foldを切るだけだと、クラス数自体は8811になるけどlabel自体は11014まである(0, 2, 3, 4, 6, 8, ...的な)ので、それを(0, 1, 2, 3, 4, ...)に直さなければいけないことだった。頑張って直した。<br>
    - その代わりとして、validationはできなくなってしまった。<br>
    - train_loss: 0.2121<br>
    - StratifiedKFoldに比べてtrain_lossが高いのはなぜなのだろうか...<br>
  - ver18~<br>
    - RTX3090で様々な損失関数を試してみた<br>
    - foldの切り方はまだstratifiedKFoldになっている。<br>
    - | ver | train_fn | train_loss | valid_loss | 
      | :---: | :---: | :---: | :---: |
      | 18 | FocalLoss(gamma=2) | 0.2074 | 1.7763 | 
      | 19 | FocalLoss(gamma=1) | 0.1516 | 1.7412 | 
      | 20 | FocalCosineLoss | - | - | 
      | 21 | SymmetricCrossEntropyLoss | 0.1125 | 0.1712 | 
      | 22 | BiTemperedLoss | - | - | <br>
    - FocalCosineLossとBiTemperedLossは挙動がおかしかったので止めた。SymmetricCrossEntropyLossはCrossEntropyLossとほぼ変わらなかった。効くとしたらFocal Lossだが、valid_lossがうまく下がらないので今回は採用しない方がいいかもしれない。<br>
  - ver23<br> 
    - ver18にDual Attention Headを入れた。<br> 
    - 一旦quick saveしている。<br>
  - ver24<br>
    - RTX 3090で動かした。<br> 
    - train_loss: 0.1740<br>
  - ver25<br>
    - GroupKFoldでFocalLoss(gamma=0.5)を試した。<br>
    - 

- nb003<br>
  - ver18(ver17は失敗)<br>
    - | train_ver | CV | LB | 
      | :---: | :---: | :---: |
      | 17 | 0.7444 | 0.716 | <br>
    - とりあえずGroupKFoldがしっかり機能してくれたことに安心。ここから何度かこのやり方でCVとLBを観察する必要がありそう。<br>
  - ver19<br>
    - | train_ver | CV | LB | 
      | :---: | :---: | :---: |
      | 18 | 0.7562 | - | <br>
    - Foldの切り方はStriatifiedになっている。<br>
  - ver20<br> 
    - | train_ver | CV | LB | 
      | :---: | :---: | :---: |
      | 19 | 0.7579 | 0.712 | <br>
    - Foldの切り方はStriatifiedになっている。<br>
    - やはり、CVとLBの相関がよくない。RANZCRでは、FocalLossはCVが下降してもLBが実は一番良かったので、GroupKFoldでも様子を見た方が良さそう。<br>
  - ver21, ver22<br>
    - Dual Attention Headのモデルを試した。その時、oofのみを使うか、全データを使うかでthresholdの値が大きく違ったので、両方試してみた。<br>
    - | train_ver | threshold | CV(oof) | CV(all data) | LB |
      | :---: | :---: | :---:| :---: | :---: |
      | 24 | 11.2 | 0.7330 | - | 0.706 | 
      | 24 | 13.2 | 0.7211 | 0.7941 | 0.667 |<br>
    - threshold自体はやはりoofに従うのが良さそう。あとはoofのCVのLBに対する相関を見ればいいだけ。<Br>
