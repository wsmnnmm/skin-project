// pages/profile/profile.js
Page({
  /**
   * 页面的初始数据
   */
  data: {
    userInfo:null
    },
  login(){
    wx.navigateTo({
      url:'//pages/log/log',
    })
  },
  tuichu(){
    this.setDaata({
      userInfo:null,
    })
    wx.setStorageSync('user2', null)
  },
  goChange(){
    wx.navigateTo({
      url: '/pages/history/history',
    })
  },
  onShow(){
    var user=wx.getStorageSync('user2')
    console.log('me----',user)
    if(user&&user.name){
      this.setData({
        userInfo:user,
      })
    }
  },
  /**
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