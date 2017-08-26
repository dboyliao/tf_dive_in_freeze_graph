## Tensorflow 與深度學習: 不太深的 Serialization

神經網路是近幾年非常熱門的機器學習演算法，除了在學術研究上有許多突破之外，

許多團隊也很樂於分享許多訓練好的模型參數以供使用。在現在，如果是簡單常見的

機器學習問題，往往不需要自己重新準備資料重新訓練模型了。


在我這次的 Tensorflow 的使用經驗分享裡，我會著重於如何把網路上常見的一些較

為鬆散的模型檔 (譬如所有參數的 npy 檔或 check points 檔案) 重新打包成可以

直接載入的 protobuf 檔。我會以實作一個 AlexNet 為例。除此之外，我也會以一

些簡單的 demo code ，稍稍分析一下底下的運作機制。希望透過這簡單的分享，讓

大家更容易上手 Tensorflow 。


Keywords: **Tensorflow**, **Serialization**, **ProtocolBuffer**


## About Me

Dboy Liao, 一個喜歡寫程式算算數學的業餘碼農。

- <qmalliao@gmail.com>
- [FB](https://www.facebook.com/dboyliao)
