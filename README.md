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
    - train_loss: 0.1206<br>
  - ver26<br>
    - ver17から、margin=0.5にしてepoch=15に変更した。<br>
    - train_loss: 8.0259<br>

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
  - ver22, ver23(ver21は失敗)<br>
    - Dual Attention Headのモデルを試した。その時、oofのみを使うか、全データを使うかでthresholdの値が大きく違ったので、両方試してみた。<br>
    - | train_ver | threshold | CV(oof) | CV(all data) | LB |
      | :---: | :---: | :---:| :---: | :---: |
      | 24 | 11.2 | 0.7330 | - | 0.706 | 
      | 24 | 13.2 | 0.7211 | 0.7941 | 0.667 |<br>
    - threshold自体はやはりoofに従うのが良さそう。あとはoofのCVのLBに対する相関を見ればいいだけ。<br>
  - ver24<br>
    - | train_ver | CV | LB |
      | :---: | :---: | :---: | 
      | 25 | 0.7444 | - | <br>
  - ver25<br>
### 20210419<br>
- nb003<br>
  - ver25<br>
    - | train_ver | CV | LB |
      | :---: | :---: | :---: | 
      | 26 | 0.7549 | 0.716 | <br>
    - もう本当に理解し難い。YYamaさんはほぼ同じ条件でLB0.722を出しているので、YYamaさんに画像処理は一任して自分はTextベースのアプローチを試みる。<br> 
  - ver27(ver26は失敗)<br>      
    - ここにきて、nb002で、defalutでmargin=Falseにしてたのはミスだと気づいた。他のどのNotebookを見てもmargin=0.5になっている。しかも、nb003はしっかりmargin=0.5になっているので、ずっと学習と推論でmarginの値が違うまま使ってた。完全にやらかした。<br>
    - そこで、こちらのmarginもFalseにして、nb002_ver17を改めてサブしてみる。<br>
    - | train_ver | CV | LB |
      | :---: | :---: | :---: |
      | 17 | 0.7444 | 0.716 |<br>
    - 手元のスコアではmarginがあろうと無かろうと変化がない。困った...<br>

### 20210420<br>
- NemuriさんとBelugaさんのお二方とマージさせていただいた。自然言語処理担当になったので、気合を入れ直して頑張る。<br>
- nb002<br>
  - ver27<br>
    - ver26のモデルを、dropout=0.1に戻して、共有用にコメントを付け加えてquick saveした。<br>
- nb003<br> 
  - ver28<br>
    - 共有用に、ver25でdropout=0.1に戻してコメントを付け加えてquick saveした。<br>

### 20210421<br>
- 発展ディープラーニングにTransformerとBERTの実装が詳しく掲載されていた。かなりわかりやすかったので、コンペが終わったら原論文を読みつつ自分の手で実装する。<br>
- nb002<br>
  - ver28<br> 
    - ver27を実際に動かした。<br>
    - train_loss: 4.3233<br>
- nb003<br>
  - ver29<br>
    - ver28を動かした。<br>
    - | train_ver | CV | LB |
      | :---: | :---: | :---: |
      | 28 | 0.7545 | 0.715 | 
    - やはり思ったよりは上がらない。コードの中身は0.730弱出てるコードとほぼ同じなはずなので、何か見落としやバグがあると思われる。<br>
### 20210422<br>
- nb004(BERT_training)<br>
  - ver1, ver2<br>
    - sentence BERTを使っている公開Notebook([training](https://www.kaggle.com/tanulsingh077/metric-learning-pipeline-only-text-sbert), [inferencce](https://www.kaggle.com/tanulsingh077/reaching-0-612-with-text-only-shopee))を参考にしてnb002_ver27にBERTを乗せた。<br>
    - とりあえず書き終えたのでquick saveした。<br>
### 20210423<br>
- nb004<br>
  - ver3<br>
    - train_lossがうまく下がらないので、schedulerをget_linear_schedule_with_warmupからCosineAnnealingLRに変えた。また、epochも10に上げた。<br>
    - stratifiedKFoldに切り替えてvalidation_lossが下がっていることが確認できたので、GroupKFoldに戻した。<br>
    - train_loss: 0.43599<br>
### 20210424<br>
- [このディスカッション](https://www.kaggle.com/c/shopee-product-matching/discussion/226260)によると、同じlabel_groupに属するべき商品が違うlabel_groupに属していることがあるらしい。その原因として、商品のページで、同じ物なのにも関わらず違うカテゴリーで出品されていることが考えられるらしい。そこまで深く考える必要はなくモデルが処理してくれるとchrisは言っているが、終盤ではこれについて一応考える必要があるかもしれない。<br>
- nb003<br>
  - ver30<br>
    - YYamaさんのNotebookで、KNNの距離をcosineで測ってさらにthresholdを全データでベストな値から0.15下げたところでサブをすると0.715から0.730まで上がったらしいので、自分も同じようにした。<br>
    - ver28のthresholdを0.30まで落とすと、LBが0.725まで上がった。こんなに上がるのか...<br>
    - GroupKFoldはlabel_groupのリークはないため、リークが原因でthresholdが高く出ていたわけではなさそう。<br>
    - 考えられる原因としては、oofはデータ数が6500なのに対してPublicが28000, Privateが48000なので、ペアをたくさんとりすぎてしまう傾向にあることが考えられる。oofのベストの値よりは低くした方がテストデータではいい結果がでそう。<br>
  - ver31<br>
    - ver30のCVを記録しておいた。<br>
    - | train_ver | CV | LB |
      | :---: | :---: | :---: |
      | 28 | 0.7448 | 0.725 | <br>
  - ver32<br>
    - BERTを組み込んだ。<br>
    - しかし、CVが思った以上に低い。BERTの方のthresholdが0.07ということは、各データがかなり接近していることを意味している。ArcMarginを使わないとこうなるとわかったので、サブせずに記録だけとって、BERTを作り直す。<br>
    - | CNN_ver | BERT_ver | CV | LB |
      | :---: | :---: | :---: | :---: |
      | nb002_ver28 | nb004_ver3 | 0.7000 | <br>
  - ver33<br>
    - ver31のthresholdを0.3から0.25に変更した。<br>
    - | train_ver | CV | LB |
      | :---: | :---: | :---: |
      | 28 | 0.7365| 0.724 | <br>
    - LBは若干下降したが、恐らくPrivateスコアはこっちの方が良さそう。PrivateデータはPublicデータよりもさらに多いから。そしておそらくLBの最適値はこの間(0.28あたり？)にありそう。<br>
  - ver36, ver37(ver34, ver35はsubmit時にTfidfになってしまっている。)<br>
    - nb004_ver4のBERTのサブを行った。<br>
    - まずCVについて。各epochで最も良いthresholdを抜き出した。なお、CNNはver31と同じ。<br>
    - | epoch | threshold | CV | 
      | :---: | :---: | :---: |
      | 10 | 0.30 | 0.7534 |
      | 15 | 0.28 | 0.7489 | <br>
    - CVは両方上昇している。epoch10の方が結果がいいので、恐らく10以降は過学習していると思われる。<br>
    - 次に、epoch10のBERTを用いてthresholdとLBの関係を見る。<br>
    - | threshold | CV | LB |
      | 0.25 | 0.7477 | 0.716 | 
      | 0.30 | 0.7534 | 0.706 | <br>
    - CVは高いが、LBは思ったより低い。Tfidfを比べてペアをより積極的に取ってきているように見えるため、一度予測の数を可視化して分布をしっかり見るべきかもしれない。<br>
      
- nb004<br>
  - ver4<br>
    - ArcFaceを入れないと特徴量マップの距離が近くなってしまうので、ArcFaceを入れた。batch_sizeを128に上げている。<br>
    - ちなみに、lrをあげるとうまく学習できなかった。バッチサイズをあげるのが一番高速化・安定化の面でよかった。<br>
    - どこまで学習させるのがいいのかよくわからないので、epochは20にして5, 10, 15, 20は保存しておくことにした。<br>
    - | epoch | train_loss |
      | :---: | :---: | 
      | 5 | 9.2698 | 
      | 10 | 1.9805 | 
      | 15 | 0.4178 | 
      | 20 | 0.2208 | <br>
    - 10~15あたりが一番良さそう。間違いなく20まではいらない。<br>
  - ver5<br>
    - BERTはTfidfと比べてペアをたくさん取ってくる傾向があったので、少し控えめにさせるためにmarginを上げることにした。<br>
    - margin=1<br>
    - | epoch | train_loss |
      | :---: | :---: | 
      | 5 | 27.0191 | 
      | 10 | 24.2476 | 
      | 15 | 23.4128 | 
      | 20 | 23.1411 | <br>
    - marginが大きすぎて全く学習が進まなかった。もう少し調べるべきだった。<br>
  - ver6<br>
    - margin=0.6(それより大きくするとうまくいかなかった。)<br>
    - | epoch | train_loss |
      | :---: | :---: | 
      | 5 | 11.7476 | 
      | 10 | 3.5060 | 
      | 15 | 1.0147 | 
      | 20 |  | <br>
### 20210424<br>
- nb003<br>
  - ver38<br>
    - CNNやBERT, TFIDFの出力を確認するために、oofをダウンロードした。<br>
    - | Text model | CNN_threshold | Text_threshold | CV | LB | 
      | :---: | :---: | :---: | :---: | :---: | 
      | BERT | 0.3 | 0.01 | 0.6945 | - | 
      | BERT | 0.3 | 0.05 | 0.7183 | - | 
      | BERT | 0.3 | 0.10 | 0.7406 | - | 
      | BERT | 0.3 | 0.15 | 0.7311 | - | 
      | BERT | 0.3 | 0.20 | 0.7406 | - | 
      | BERT | 0.3 | 0.25 | 0.7477 | 0.716 | 
      | BERT | 0.3 | 0.3 | 0.7534 | 0.706 | 
      | Tfidf | 0.3 | 0.75 | 0.7448 | 0.725 | 
      | Tfidf | 0.38 | 0.75 | 0.7509 | - | <br>
    - 一番下のCNN_thrsholdが0.38のものは、CNNのthresholdをCVに最適化したもの。<br>
  - ver39<br>
    - BERTのthresholdを0.2にした。(0.25と0.3は再掲)<br>
    - | threshold | CV | LB |
      | :---: | :---: | :---: |
      | 0.2 | 0.7406 | 0.718 | 
      | 0.25 | 0.7477 | 0.716 | 
      | 0.30 | 0.7534 | 0.706 | <br>
    - EDAした感じでも、閾値を0.10あたりにしてようやくTfidfと予測の個数が同じぐらいになっていた。やはりかなり控え目にしてTfidfとのアンサンブルを試すべきな気がする。<br>
  - ver40<br>
    - BERTをnb004_ver6のmargin=0.6のモデルに書き換えた。<br>
    - epoch10のモデルがthreshold=0.31でCVが0.7547を出したので、こちらの方がいい感じになっているように見える。<br>
  - ver42(ver41は失敗)<br>
    - ver33で、tfidfのthredsholdを0.53まで落として、nb004_ver6のepoch5のBERTを組み込んだ。ただ、調べた感じだと、Tfidfに加えてBERTを入れてもCVは悪化しているだけだった。BERTのthresholdを0.01にしてようやく少し効き目があるぐらい。BERTの望みは薄い感じがする。tfidfのthresholdはCVがベストから0.1下がるぐらいのところを選んだ。<br>
  - ver43<br>
    - Tfidfのthresholdを下げるとCVが上がっていくことに気づいたので、ver30からTfidfの閾値を0.75->0.55に下げてサブしてみた。<br>
    - LBが0.680まで下降した。CVと　LBの相関はやはり不安定...本当は全foldで取った方がいいよね。<br>
  - ver44<br>
    - nb006通りに、BERTの予測個数が40を超えたときにTfidfの予測をBERTの予測と置き換えた。CVは若干上がっている。<br>
    - その途中で、BERTのKNNの個数を70まで上げた方がCVの値が良くなることがわかった。確かにtrainingデータの中のペアの個数は最大で51だが、テストデータの中にはもっとある可能性も否定できないので増やすべきかもしれない。<br>
    - LBは0.724だった。元々がver30のLB0.725なので、ほぼ横ばい。CVを取ってるのがたった6500個なので、やはり全foldで再調査したほうがいいっぽい。<br>
- nb006(EDA_oofdf)<br>
  - ver1<br>
    - EffcientNet, Tfidf, BERTの予測結果を可視化した。<br>
    - その結果分かったのが、やはりTfidfとBERTを比べるとBERTがかなりの個数をとってきていると言うこと。BERTの閾値を0.05ぐらいにして初めてTfidfの閾値0.75と同じぐらいの個数分布になった。BERTの閾値0.2が大体EfficientNetの閾値0.3と同じぐらいなので、BERTはかなり積極的に予測をしていることがわかる。<br>
    - OOFの中にはペアの個数が50を超えるようなデータが存在する。予測結果を見ると、30~50個取ることができているのはBERTだけなので、なんとかこれだけでも生かせればスコアが伸びそう。<br>
  - ver2<br>
    - 正解のペアの個数毎にf1スコアの平均をとってみると、30個以上においてはBERTが一番強い(というかBERTしかそれだけとってくることができない)ことがわかった。ペアが50個のデータで、EfficientNetが１個だけ、Tfidfも一個だけみたいな状況下でもBERTは50個とってきていたりするので、BERTの予測個数が多い時だけBERTに置き換えるといいと考えられる。<br>
    - チーム内の分析で、BERTは実は予測個数が2個や３個の時に明らかに強いことが判明した。実際にその置き換えを行ったところスコアが0.005伸びたので、これは効きそう。<br>
### 20210426<br>
- nb004<br>
  - ver11(ver7, 8, 9, 10は失敗)<br>
    - ver6のBERTを全foldで作った。10epochが一番よかったため、T_maxを20にしたままepoch10で切ることにした。<br>
    - fold0を飛ばすコードを書いたままにしてしまっていた。quotaが余りそうなのでもう一度やり直す。<br>
  - ver12<br>
    - 今度は余計なモデルまでセーブしてしまってoutputが膨大になってしまった。outputのうち選択的に選んでデータセットにする方法ってないのか...？<br>
  - ver13<br>
    - やっと成功した。<br>
    - | fold | train_loss | 
      | :---: | :---: | 
      | 0 | 3.5060 |
      | 1 | 3.5618 | 
      | 2 | 3.6370 | 
      | 3 | 3.3603 | 
      | 4 | 3.5061 | <br>
    - stratifiedで回してみたところ、validation_lossは14あたりから下がり方がかなり鈍くなってきて、12あたりが限界だった。それ以降は過学習していることは頭に入れなければいけない。<br>
### 20210427<br>
- nb002<br>
  - ver29, 30<br>
    - ver28のモデルを全foldに関して作った。<br>
    - ver29にfold1~4, ver30にfold0のモデルがある。<br>
    - lrをバッチサイズに対して線形に変化させているが、バッチサイズが大きくなるとtrain_lossがより下がってしまうのが少し気になる。これまでのコンペとは違い理論値通りにいかないが、validation_lossも出せないし気にしててもしょうがない気もする。<br>
- nb003<br>
  - ver45<br>
    - BERTの予測個数が2個の場合にTfidfの出力をBERTの出力で置き換える処理をした時のCVの変化を記録した。<br>
  - ver46<br>
    - nb002_ver29, ver30のモデルを入れた。CVを出す処理はまだ書いていない。<br>
    - LBが0.733まで上がった！やはり全foldを入れるだけでも安定感が違う。<br>
- nb005<br>
  - ver1, ver2<br>
    - EffNetとBERTとTfidfを合わせるNotebookをnb003と切り離した。<br>
    - nb002_ver29, 30とnb004_ver13のモデルでCVを出そうとしたが、modelを何個もロードしているうちにRAMに収まりきらなくなってしまう。del modelの後にgc.collect()もしているので消えていると思っていたがどうも消えてないらしい。正直どうしてかわからない。循環参照があるようにも見えない。torch.cuda.empty_cache()も試してみたが、そもそもcpu側の問題なのでどうにもならないっぽい。<br>
### 20210429<br>
- nb003<br>
  - ver47<br>
    - 全foldを使ってCVを出せるようにコードを色々書いてみたが、まだ書き終わっていない。image_embeddingsとtext_embeddingsを元の順番に戻すのに大苦戦してる。<br>
  - ver48<br>
    - Belugaさんが組んでくれた補完アルゴリズムを入れて、さらに置き換えをTfidfの予測値が一個の場合に切り替えた。<br>
    - LBが0.734->0.724まで下降した。Tfidfが一個の場合に置き換えるのはそもそもSEResNeXtを対象に行なった実験であるから、そのせいなのかもしれない。こっちでも出力を解析する必要がありそう。<br>
  - ver49<br>
    - Nemuriさんが作ってくれた前処理付きBERTに差し替えて実験をした。それ以外はLB0.734の時と同じ。<br>
    - 0CVは0.8001->0.8005でLBは0.734->0.733に落ちてしまった。この辺は正直閾値に依存している部分もあるのであんまり気にしなくていいと思う。<br>
    - あと、あまり本質的ではないけどバッチサイズは8から32に上げても大丈夫だった。次からそうする。<br>
- nb006<br>
  - ver3, ver4<br>
    - nb006のoofを解析した。すると、一番CVがよくなるのは、Tfidfの出力が１個の場合にBERTで置き換えた時だった。考えてみれば当然で、ペアは２つ以上あるので、１個は必ず間違っているからだろう。<br>
    - pred_matchesが１つしかないデータの個数はは535->492に減った。置き換えをしても、依然として明らかに誤っているデータは残されているため、そこは明示的にアルゴリズムを組んで補完した方がいいと思われる。それにあたって、[このディスカッション](https://www.kaggle.com/c/shopee-product-matching/discussion/233626)を参考にしたい。<br>
- nb007(training_SResNeXt)<br>
  - ver1<br>
    - nb002_ver29を全foldで回せるように修正してモデルをSeResNeXtに書き換えた。<br>
    - Kaggle Notebookのsaveを失敗してそっちではver2になってしまった。<br>
- nb008(infer_SeResNeXt)<br>
  - ver1, ver2, ver3, ver4<br>
    - nb003を参考に、ひとまずfold0のみで、BERTの予測個数が2の場合にTfidfの出力と入れ替える処理を書いた。<br>
    - oof_dfをpickle形式で保存できるようにした。今回の出力はリストになっているため、csvファイルで保存するとstrとして読み込まれてしまう。pickleファイルにすればそのまま読み込めるので便利。<br>
  - ver5<br>
    - 比較をしやすくするために、BERTのthresholdを0.2に下げた。<br>
  - ver6<br>
    - cnn_threshを0.4, BERTのthreshを0.3にしてサブした。Belugaさんのマッチ補完関数を入れるのを忘れてた。BERTを全fold分入れることができないため、fold0のみでCVを出している<br>
    - | CV | LB |
      | :---: | :---: | 
      | 0.8227 | 0.724 | <br>
    - 結構乖離が大きい。CVに対するbestは0.54だったが、データ数的にもっと思いっきり減らすべきだったか。<br>
  - ver7<br>
    - サブが余ったので、閾値を0.35に落として、マッチ補完関数も入れてサブした。<br>
    - | CV | LB |
      | :---: | :---: |
      | 0.8140 | 0.731 | <br>
    - だいぶ近づいた。おそらくまだ伸びる。BERTも正直もっと下げた方がいいはずなので、サブが余った時に試してみたい。<br>
### 20210430<br>
- ここ数日のものを全て20210427に書いてしまった。まあいいけど。<br>
- nb003<br>
  - ver50<br>
    - BelugaさんのLB0.734のNotebookを整理して、BERTのthreshfoldを0.3->0.2に落とした。過去にBERTだけでやった時は0.3より0.2の方がスコアが良かったので改善されるかもしれない。<br>
    - | CV | LB |
      | :---: | :---: |
      | 0.7913 | 0.732 | <br>
    - このBERTの使い方では、thresholdはいじらない方がいいっぽい。<br> 
  - ver51<br>
    - CNNの予測値を、一個しかなければ必ずもう一つ取らせるようにコードを変えた。正解は必ず二つ以上なので(ペアは必ず存在するから)、改善される可能性がある。<br>
    - サブするのはやめた。<br>
### 20210501<br>
- nb002<br>
  - ver32, ver33<br>
    - ver28から、ArcMarginProductを書き換えてMultiFaceにした。論文を読んで真似しただけなので間違ってるかもしれない。<br>
    - ver32がepoch15, ver33がepoch30になっている。<br>
    - | epoch | train_loss | 
      | :---: | :---: |
      | 15 |  | 4.6513 | 
      | 30 | 4.0297 | <br>
    - かなりいい感じでは？<br>
  - ver34<br>
    - fold1~4についても回した。<br>
- nb003<br>
  - ver56(ver52, ver53は失敗)<br>
    - 予測値が一個の場合に一番近い予測値をとってくる処理を付け加えた。<br>
    - 間違えてcnn_threshが0.25のままでサブしてしまった。ver54はこれのquick save。<br>
  - ver57<br>
    - 予測値が一個の場合にもう一つとってくる処理をCNNだけにした。<br>
    - これも間違えてcn_threshが0.25になっている。ver55はこれのquick save。<br>
  - ver58<br>
    - ver56のcnn_threshを0.3に直した。<br>
    - | CV | LB |
      | :---: | :---: | 
      | 0.8119 | - | <br>
  - ver59<br>
    - ver57のcnn_threshを0.3に戻した。<br>
    - | CV | LB |
      | :---: | :---: | 
      | 0.8114 | - | <br>
  - ver60<br>
    - nb002_ver33のCVを算出した。<br>
    - nb002_ver28のCVが0.7996. nb002_ver33のCVが0.7974で若干下降している。これだけで却下するのは若干心許ない。データ数が少ない故かもしれないで、全て作り切ってみた方がいいかもしれない。<br> 
