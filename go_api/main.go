package main

import (
	"bytes"
	"encoding/json"
	"github.com/gin-gonic/gin"
	"io/ioutil"
	"net/http"
	"path/filepath"
)

type RequestBody struct {
	Prompt string `json:"prompt"`
}

type ResponseBody struct {
	Response string `json:"response"`
}

func main() {
	r := gin.Default()
	r.LoadHTMLGlob(filepath.Join("templates", "*.html"))
	r.Static("/static", "./static")

	// 网页入口
	r.GET("/", func(c *gin.Context) {
		c.HTML(200, "index.html", nil)
	})

	// API 调用（支持页面 JS 和 Postman 调用）
	r.POST("/infer", func(c *gin.Context) {
		var req RequestBody
		if err := c.BindJSON(&req); err != nil {
			c.JSON(400, gin.H{"error": "Invalid request"})
			return
		}

		// 请求 Python 后端
		reqJson, _ := json.Marshal(req)
		resp, err := http.Post("http://localhost:6006/infer", "application/json", bytes.NewBuffer(reqJson))
		if err != nil {
			c.JSON(500, gin.H{"error": "调用 Flask 服务失败", "detail": err.Error()})
			return
		}
		defer resp.Body.Close()

		body, _ := ioutil.ReadAll(resp.Body)
		var result ResponseBody
		if err := json.Unmarshal(body, &result); err != nil {
			c.JSON(500, gin.H{"error": "返回格式异常", "raw": string(body)})
			return
		}

		c.JSON(200, result)
	})

	r.Run(":8080")
}
