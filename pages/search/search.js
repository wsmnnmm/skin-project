// pages/search/search.js
Page({
  /**
   * 页面的初始数据
   */
  data: {
    imagePath:'/image/skin.jpg'
  },
chooseImage(e){
  const that = this;
  wx.chooseMedia({
    success:(res)=>{
      //获取图片的临时路径
      const tempFilePath = res.tempFiles[0].tempFilePath
      that.setData({
        imagePath:tempFilePath
      })
      var testName = this.data.imagePath.split('/')[3]||this.data.imagePath.split('/')[2];
      
      that.setData({
        imageName:testName,
      });
      //根据官方的要求 用Base64字符编码获取图片的内容
      wx.getFileSystemManager().readFile({
        filePath:tempFilePath,
        encoding:'base64',
        success:function(res){
          //返回base64格式
          //console.log('data:image/png;base64,'+res.data)
          that.setData({
            picture:res.data
          })
        }
      })
    }
  })
},

doClickWork(){
  var that = this;
  if(!that.data.picture || !that.data.imageName){
    wx.showModal({
      title:'请选择图片！'
    })
    return;
  }

  wx.showLoading({
    title:'正在上传...',
  })

  wx.request({
    url: 'http://127.0.0.1:5000/predict',//本地服务器地址
    method:'POST',
    header:{
      'content-type':'application/x-www-form-urlencoded'
    },

    data:{
      'picture':that.data.picture,
    },
    success: (res)=>{
      that.setData({
        class_id:res.data['class_id'],
        class_name:res.data['class_name'],
        prob:res.data['prob']
      })

    }

  })
},

  /**
   * 
   * 生命周期函数--监听页面加载
   */
  onLoad(options) {

  },

  /**
   * 生命周期函数--监听页面初次渲染完成
   */
  onReady() {

  },

  /**
   * 生命周期函数--监听页面显示
   */
  onShow() {

  },

  /**
   * 生命周期函数--监听页面隐藏
   */
  onHide() {

  },

  /**
   * 生命周期函数--监听页面卸载
   */
  onUnload() {

  },

  /**
   * 页面相关事件处理函数--监听用户下拉动作
   */
  onPullDownRefresh() {

  },

  /**
   * 页面上拉触底事件的处理函数
   */
  onReachBottom() {

  },

  /**
   * 用户点击右上角分享
   */
  onShareAppMessage() {

  }
})