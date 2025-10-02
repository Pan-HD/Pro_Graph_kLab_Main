### Setting - VisualStudio - OpenCV

---

- P1 - 视图 - 其他窗口 - 属性管理器

- P2 - release x64 - 添加新项目属性表

  - `Microsoft.Cpp.x64.user`

- P3 - Microsoft.Cpp.x64.user - 属性

  - VC++

    - 包含目录

      - ```
        D:\Program Files (x86)\opencv_4.9.0\opencv\build\include
        D:\Program Files (x86)\opencv_4.9.0\opencv\build\include\opencv2
        ```

    - 库目录

      - ```
        D:\Program Files (x86)\opencv_4.9.0\opencv\build\x64\vc16\lib
        ```

  - 链接器 - 输入 - 附加依赖项

    - ```
      opencv_world490.lib
      ```

- P4 ( .dll文件相关报错解决 )

  - ```
    把 opencv_world490.dll 复制到 Pro_Graph_kLab.exe 所在的目录: 
    F:\ComputerScience\OpenCV_VS_Pro\Pro_Graph_kLab\x64\Release
    ```

