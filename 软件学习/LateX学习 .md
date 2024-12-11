# LateX学习

[免费网站](https://www.latexstudio.net/LearnLaTeX/lesson/01.html) 

[**CTAN**](https://www.ctan.org/) 全称 Comprehensive TeX Archive Network（TeX 综合资料网）。绝大多数的 LaTeX 宏包都会在这里发布，因此你也可以通过在该网站搜索以访问其文档。一般来说，宏包会位于 `ctan.org/pkg/<pkg-name>`路径下，在这里通常可以找到相应的 README 和文档。

[**latex模版库**](http://www.latextemplates.com/cat/presentations)

[大学模版库](https://github.com/supresu/Latex-Templates) 

## 1.基础逻辑格式

每个 LaTeX 文档都有一个 `\begin{document}` 和一个匹配的 `\end{document}`。在这两者之间是文档的**主体**，即你的内容所在。这里的正文有两个段落（在 LaTeX 中，你用一个或多个空行来分隔段落）。

**在 `\begin{document}` 之前是文档导言区（preamble），其中有设置文档布局的代码。**

`\usepackage` 命令将在后面的课程中介绍，在本网站的大多数例子中，它被用来设置字体编码。

LaTeX 还有其他 `\begin{...}` 和`\end{...}` 的搭配；我们称这些为**环境（environments）**。你必须正确匹配它们，以便每一个 `begin{x}` 都有一个 `end{x}`。如果你对它们进行嵌套，那么你必须有`\end{y} ... \end{x}` 来匹配 `\begin{x} ... \begin{y}`，即按 `\begin` 和 `\end` 语句的顺序匹配。

在 LaTeX 文件中添加以 `%` 开头的注释

```latex
\documentclass[a4paper,12pt]{article} % 使用选项的的文档类
\usepackage[T1]{fontenc}
% 在导言区里的注释
\begin{document}
% 这是一个注释
This is   a simple
document\footnote{with a footnote}.

This is a new paragraph.
\end{document}
```

以 `\` 起始是 LaTeX 指令：一个「命令」。大括号字符 `{` 和 `}` 用于显示**强制性参数（mandatory arguments）**：命令需要的信息。

### 1.1 宏包扩展

/声明一个文档类（文档类在后续提到，具体是指 \documentclass{} ）后，你在 LaTeX 导言区中可以通过添加一个或多个「宏包（package）」来修改相关功能。宏包可以：

- 改变 LaTeX 某些部分的功能
- 向 LaTeX 添加新的命令
- 更改文档的设计

对于用户来说，自定义是很受限制的，所以需要使用扩展包来实现一些常见功能。第一件事就是在 LaTeX 中如何改变对于特定语言的排版方式（断字、标点符号、引文、本地化……）。因为不同的语言有不同的规则，所以告诉 LaTeX 要使用哪种规则是很重要的。这可以通过 `babel` 宏包来处理。

`\usepackage` 命令接受宏包逗号分隔列表作为参数，所以你可以在一行中加载多个宏包：比如，`\usepackage{color,graphicx}`。传递的参数会向该列表中所有的宏包传递。分别加载的话，注释掉宏包也会更简单，因此我们会保持一行加载一个宏包的这个习惯。

```latex
\documentclass{article}
\usepackage[T1]{fontenc}

%\usepackage[french]{babel}

\usepackage[width = 6cm]{geometry} % 强制进行断字

\begin{document}

This is a lot of filler which is going to demonstrate how LaTeX hyphenates
material, and which will be able to give us at least one hyphenation point.
This is a lot of filler which is going to demonstrate how LaTeX hyphenates
material, and which will be able to give us at least one hyphenation point.

\end{document}
```

调整某些独立于文档类方面的设计通常是有用的。最常见的是更改页边距。刚刚就在上面的例子中使用了 `geometry` 宏包，让我们现在再来看一个专门关于页边距的例子。

1.1.1 更改页边距

调整某些独立于文档类方面的设计通常是有用的。最常见的是更改页边距。刚刚就在上面的例子中使用了 `geometry` 宏包，让我们现在再来看一个专门关于页边距的例子。

```latex
documentclass{book}
\usepackage[T1]{fontenc}
\usepackage[margin=1in]{geometry} %改变页边距

\begin{document}
Hey world!

This is a first document.

% ================
\chapter{Chapter One}
Introduction to the first chapter.

\section{Title of the first section}
Text of material in the first section

Second paragraph.

\subsection{Subsection of the first section}

Text of material in the subsection.

% ================
\section{Second section}

Text of the second section.

\end{document}
```

1.1.2 更改颜色的包

为了设定彩色加载了 `xcolor` 宏包，在格式上把加粗改成了蓝色。

```latex
\documentclass{article}
\usepackage[T1]{fontenc}

\usepackage{xcolor}

\newcommand\kw[1]{\textcolor{blue}{\itshape #1}}

\begin{document}

Something about \kw{apples} and \kw{oranges}.

\end{document}
```

## 2. 基础代码和格式

`\emph`：将文字变成斜体

`\textbf` ：让文字变粗的命令

```latex
\documentclass{article}
\usepackage[T1]{fontenc}
\begin{document}
Some text with \emph{emphasis and \emph{nested} content}.

Some text in \textit{italic and \textit{nested} content}.
\end{document}
```

`\section` ：它不仅可以直接控制字体变化、纵向间距……，还可以让整个文章的格式统一

```latex
\documentclass{article}
\usepackage[T1]{fontenc}
\begin{document}
Hey world!

This is a first document.

\section{Title of the first section}

Text of material in the first section

Second paragraph.

\subsection{Subsection of the first section}

Text of material in the subsection.

\section{Second section}

Text of the second section.

\end{document}
```

`\article` 配置时，LaTeX 会对节与小节进行编号，并把标题加粗。

LaTeX 可以将文档分成好几个层级：

- `\chapter` （启用它需要 `\documentclass{book}` 或者 `\documentclass{report}`）
- `\section`
- `\subsection`
- `\subsubsection`

更进一步来讲，再下一个层级就是 `\paragraph`，但是基本上这一层就会对目次划分得「太细」。（没错，`\paragraph` 是一个目次命令，**不是**开始新段落的命令！）

 `\item` 来开始每一项，这样每一种列表所使用的标记或编号就会自动递增。表示列表

**\begin{enumerate}中的参数enumerate表示递增编号，而itemize表示button点标记**

```latex
\documentclass{article}
\usepackage[T1]{fontenc}
\begin{document}

Ordered
\begin{enumerate}
  \item An entry
  \item Another One
  \item Wow! Three entries
\end{enumerate}

Unordered
\begin{itemize}
  \item An entry
  \item Another One
  \item Wow! Three entries
\end{itemize}

\end{document}
```

## 3.文档类

文档类设置了文档的一般布局，比如：

- 设计边距、字体、间隔……
- 是否会有章这一级
- 标题是否需要另起一页

更一般来讲，文档类也可以添加新的命令：特别是专用情形，比如创建幻灯片时。

文档类这一行也可以被设置一些**全局选项**——对整个文档起作用的选项。这些选项需要在方括号中给出：`\documentclass[<选项>]{<文档类>}`。许多 LaTeX 命令都会使用这种首先在方括号中给出可选信息的语法。

### 3.1 基本文档类

- `article`不包含章的短文档
- `report`含有章的长文档，单面印刷
- `book`含有章的长文档，双面印刷，包含正文前材料（front-matter）和正文后材料（back-matter，比如索引）
- `letter`不包含目次信息
- `slides`幻灯片（详见后文）

`article`, `report`, `book` 文档类都有非常相似的命令。当撰写一个 `letter` 时，可以使用的命令已经有点不同了。

```latex
\documentclass{letter}
\usepackage[T1]{fontenc}
\begin{document}

\begin{letter}{Some Address\\Some Street\\Some City}

\opening{Dear Sir or Madam,}

The text goes Here

\closing{Yours,}

\end{letter}

\end{document}
```

### 3.2 富功能文档类

两个最大且最流行的扩展类是 KOMA-Script 包和 memoir 文档类。KOMA-Script 包提供了标准类的一些「平行」类：`scrartcl`, `scrreprt`, `scrbook` 以及 `scrlttr2`，而 `memoir` 文档类更像是 `book` 类的一种拓展。

（CTeX 中文社区也提供了 ctex 宏包，其中包含了 `ctexart`, `ctexrep` 和 `ctexbook` 三个文档类，用来编写中文短文、中文报告和中文书籍。）

### 3.3 幻灯片演示

`slides` 是为 20 世纪 80 年代中期物理投影片开发的文档类

`slides` 文档类是为印制传统幻灯片编写的，对于屏幕演示没有任何特殊的支持。两种文档类被开发出来用于后者的用途、并被广泛使用：`beamer` 和 `powerdot`。因为 `beamer` 可能是更加常见的一种

```latex
\documentclass{beamer}
\usepackage[T1]{fontenc}
\begin{document}

\begin{frame}
  \frametitle{A first frame}
  Some text
\end{frame}

\begin{frame}
  \frametitle{A second frame}
  Different text
  \begin{itemize}
    \item<1-> First item
    \item<2-> Second item
  \end{itemize}
\end{frame}

\end{document}
```

这里展示了两个重要的想法。首先，`beamer` 将文档分割成帧（frame），每一帧都可以产生多于一个幻灯片（页）。其次，`beamer` 向普通的 LaTeX 语法添加功能以允许代码中的一些部分「每次显示一点儿」

### 3.4 组建更长的文档

编写长文档的时候，很可能想将源代码分割成多个文件。比如，很常见的做法是，创建一个 `main` 或 `root` 文件，然后每一章创建一个源文件（对于一本书或者一篇论文），或者每一个长节创建一个源文件（对于一篇长文档）

LaTeX 允许我们以可控的方式分割源代码。对此，有两个很重要的命令：`\input` 和 `\include`。`\input` 让文件「看起来就是输入在这里一样」，所以（几乎）可以用于任何文档。

`\include` 命令仅对章节有用：从新页开始并做一些内部调整。但它有一个很大的优势：允许我们选择输入哪些章节，这样就可以不用排印全文而只需要排印这一部分的内容。

3.4.1 使用input

`\input` 命令适合于**不**独立于章节的长文件片段。例如，我们用它来分隔前封面和后封面，来保持主文件的简短清晰，并可以在其他文档中复用。我们也用这个命令来分割书的起始「非章节」目次：比如前言。这也是为了保持主文件的结构清晰。

3.4.2 使用include和includeonly

对于章节而言，`\include` 很有用，因此我们对于每一个整章都使用了这个命令。我们通过使用 `\includeonly` 来选择哪一章会被实际排印出来（这个命令接受逗号分隔的文件列表）。当你使用 `\includeonly` 时，可以缩短排版出来的文件长度并制作出一个「选择后的」PDF 用于校对。而且，`\includeonly` 的核心优势在于 LaTeX 会使用其他所有包含进来文件的 `.aux` 辅助文件用于章节间的交叉引用。

3.4.3 创建目录

`\tableofcontents` 命令使用目次命令的信息来制作目录。它有着以 `.toc` 为后缀的专门辅助文件，所以你可能需要运行 LaTeX 两次以解析这些信息。目录将会从目次标题自动生成。还有一些类似的命令比如 `\listoffigures` 以及 `\listoftables`，将根据浮动体环境的标题分别生成以 `.lof` 和 `.lot` 后缀的文件。

3.4.4 分割文档

`\frontmatter`, `\mainmatter` 和 `\backmatter` 命令影响着格式。例如，`\frontmatter` 改变页码为罗马数字。`\appendix` 命令将改变章节编号为 `A`, `B`

### 3.5 字体

因为 `pdflatex` 被限制为 8 位文件编码和 8 位字体，所以它并不能原生地支持现代 OpenType 字体（使用不同的字母与用于输入专业术语的脚本）来在不同的语言之间切换。当下 pdfTeX 有两个可原生使用 Unicode 输入与现代字体的替代引擎：XeTeX 和 LuaTeX。对于 LaTeX 版本而言，这些引擎通常在你的编辑器中分别通过 `xelatex` 和 `lualatex` 调用。

在这些引擎中，字体选择是通过 `fontspec` 宏包实现的。对于简单的文档可以如下面的例子一样简单：

```latex
\usepackage{fontspec}\setmainfont{texgyretermes-regular.otf}
```

这个例子选择了 TeX Gyre Termes 字体，和上面的 `pdflatex` 例子一样。值得注意的是，这种方法对 *任何*OpenType 字体都能用。一些对 `pdflatex` 可用的字体在 `xelatex` 和 `lualatex` 中通过对应的宏包也可以使用。或者是和上面一样，通过使用 `fontspec` 宏包加载你电脑上安装的任何字体。

## 4 插图

### 4.1 插图

为了向 LaTeX 文档插入外部来源的图片，需要使用 `graphicx` 宏包来向 LaTeX 添加 `\includegraphics` 命令。

```latex
\documentclass{article}
\usepackage[T1]{fontenc}
\usepackage{graphicx}

\begin{document}
This picture
\begin{center}
  \includegraphics[height=2cm]{example-image}
\end{center}
is an imported PDF.
\end{document}
```

可以插入 EPS, PNG, JPG 以及 PDF 文档

### 4.2 图片修改

`\includegraphics` 命令为调整图片大小和形状、裁切图片提供了许多选项。其中有些选项是很常用的，需要稍微关注一下。

最常见的就是设定一个图片的宽度和高度，通常被设定为 `\textwidth`（或 `\linewidth`） 和 `\textheight` 的相对值。这里 `\textwidth` 和 `\linewidth` 之间的差别是很微妙的，并且通常是相同的。`\textwidth` 是一整页的文本区域宽度，而 `\linewidth` 是当前行的宽度，会因为不同的行而不同（当启用文档类中的 `twocolumn` 选项时这种差别会非常明显）。LaTeX 会自动缩放图片以保持正确的宽高比例。

```latex
\documentclass{article}
\usepackage[T1]{fontenc}
\usepackage{graphicx}

\begin{document}
\begin{center}
  \includegraphics[height = 0.5\textheight]{example-image}
\end{center}
Some text
\begin{center}
  \includegraphics[width = 0.5\textwidth]{example-image}
\end{center}
\end{document}
```

也可以尝试采用 `scale` 命令缩放图片，或者指定 `angle` 角度以旋转图片，亦可通过 `clip` 和 `trim` 裁切图片。

```latex
\documentclass{article}
\usepackage[T1]{fontenc}
\usepackage{graphicx}

\begin{document}
\begin{center}
  \includegraphics[clip, trim = 0 0 50 50]{example-image}
\end{center}
\end{document}
```

### 4.3 浮动图片

图片通常会被设定为浮动体，从而页面中不会有大面积的留白。

```latex
\documentclass{article}
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\usepackage{lipsum}  % 产生填充假文

\begin{document}
\lipsum[1-4] % 只是一些填充文段

Test location.
\begin{figure}[ht]
  \centering
  \includegraphics[width=0.5\textwidth]{example-image-a.png}
  \caption{An example image}
\end{figure}

\lipsum[6-10] % 只是一些填充文段
\end{document}
```

`ht` 选项影响了 LaTeX 在何处放置浮动体：这两个字母的意思是，浮动体可以在原来的地方（`Test location` 旁边），或者是一页的顶部。你最多可以使用四种位置描述符（position specifier）：

- `h` Here 这里（如果可以的话）
- `t` Top 页面顶部
- `b` Bottom 页面底部
- `p` Page 浮动体专页

使用 `\centering` 而不是 `center` 环境来水平居中图片。这将避免浮动环境和 `center` 环境都会增加纵向间隔的局面。

## 5 表格

[表格进阶及查询](https://www.latexstudio.net/LearnLaTeX/more/08.html)

### 5.1 表格插入

在 LaTeX 中，使用 `tabular` 环境构建表格。本课会假设你已经加载了 `array` 宏包以向 LaTeX 表格添加更多的功能。在导言区添加下面的代码即可继续操作：

```latex
\usepackage{array}
```

为了排版 `tabular` 表格，我们需要告诉 LaTeX 总共有多少列，以及应当怎样对齐。这通常通过一个额外的参数——通常被称为表格导言（table preamble）——来指定 `tabular` 列数。每列通常通过单个字母（被称为引导符，preamble-token）指定。可选的列格式如下：

| 类型 | 描述 |
| --- | --- |
| `l` | 列左对齐 |
| `c` | 列居中对齐 |
| `r` | 列右对齐 |
| `p{width}` | 固定列宽；文字会被自动折行并两端对齐 |
| `m{width}` | 和 `p` 类似，但垂直居中对齐 |
| `b{width}` | 和 `p` 类似，但垂直底部对齐 |
| `w{align}{width}` | 固定列宽，如果太长会稍稍出界。你可以选择水平对齐（align）选项 `l`, `c` 或 `r` |
| `W{align}{width}` | 和 `w` 类似, 但是如果出界的话会收到警告 |

被 `l`, `c`, `r` 标识的列将会根据最宽的单元格自动决定列宽。每一列都需要被声明。如果需要三个居中列，你可以在表格导言使用 `ccc`，当然，因为空格会被忽略掉，`c c c` 也是等同的。

表格主体中，列都是通过 `&` 和号来分隔的，行是通过 `\\` 来另起的。

```latex
\documentclass{article}
\usepackage[T1]{fontenc}
\usepackage{array}

\begin{document}
\begin{tabular}{lll}
  Animal & Food  & Size   \\
  dog    & meat  & medium \\
  horse  & hay   & large  \\
  frog   & flies & small  \\
\end{tabular}
\end{document}
```

如果一个表格列包含了太多的文字，仅使用 `l`, `c` 和 `r` 的话你可能会遇到麻烦。

造成这个问题的原因是，`l` 类型的列即使已经超出了页面的范围，也会将所有的内容排版成一行。为了解决它，你可以采用 `p` 类型。这种类型将内容排版为指定宽度的段落，垂直顶部居中

如果你的表格里有太多列是同样的类型，那么向表格导言重复写入大量描述符就太麻烦了。你可以通过使用 `*{num}{string}` 让事情变得简单一些，这会让 `string` 格式描述符重复 `num` 次。所以 `*{6}{c}` 和 `cccccc` 是等价的。

### 5.2 行分割线

表格中应当少用分割线，而且纵向分割线一般看起来不专业。事实上，在专业表格中，你不应当使用任何标准分割线；取而代之的，你应该熟练使用 `booktabs` 宏包里提供的工具，因此我们这里打算先讨论它。

`booktabs` 提供了四种不同的分割线。每一种命令都需要在每一行之前或者在一个分割线之后使用。其中三种分割线命令是：`\toprule`, `\midrule`, `\bottomrule`。

```latex
\documentclass{article}
\usepackage[T1]{fontenc}
\usepackage{array}
\usepackage{booktabs}

\begin{document}
\begin{tabular}{lll}
  \toprule
  Animal & Food  & Size   \\
  \midrule
  dog    & meat  & medium \\
  horse  & hay   & large  \\
  frog   & flies & small  \\
  \bottomrule
\end{tabular}
\end{document}
```

第四种 `booktabs` 提供的分割线命令是 `\cmidrule`，可用来绘制出仅占用指定列范围、而不占满整行的分割线。列范围被表示为一个数字范围：`{`*列号*`-`*列号*`}`。即使你只需要对一列画分割线，也要指定一个范围（范围两端的列号相同罢了）。

有时，一条分割线对于分割两行来说可能做的还是太过了，希望通过其他方式将两行更清晰地分隔开。在这种情况下，可以使用 `\addlinespace` 来插入一个小的间隙。

### 5.3 合并单元格

在 LaTeX 中，你可以通过 `\multicolumn` 命令水平合并两个单元格，如果需要使用这个命令，必须在输入单元格内容前使用。`\multicolumn` 接受三个参数：

1. **需要合并多少个单元格**
2. **合并后单元格的对齐方式**
3. **合并后单元格的内容**

对齐方式可以使用 `tabular` 中定义的任何方式，但只需要指定**一个**（不是多个）列类型。

```latex
\documentclass{article}
\usepackage[T1]{fontenc}
\usepackage{array}
\usepackage{booktabs}

\begin{document}
\begin{tabular}{lll}
  \toprule
  Animal & Food  & Size   \\
  \midrule
  dog    & meat  & medium \\
  horse  & hay   & large  \\
  frog   & flies & small  \\
  fuath  & \multicolumn{2}{c}{unknown} \\
  \bottomrule
\end{tabular}
\end{document}
```

你也可以在单个单元格上使用 `\multicolumn` 来屏蔽表格导言中对该列的定义。使用下面的方法居中表头。

LaTeX 不支持纵向单元格的合并。通常通过将单元格留空的方式，来告诉读者单元格是跨行的。

## 6 交叉引用

在编写任何长度的文档时，你可能都想引用一些编号过的项目，例如：图片、表格或者公式。幸运的是，我们只需要进行些许设置，LaTeX 便能够自动添加正确的编号。

为了让 LaTeX 记住文档中的一个位置，你必须先标记它，然后在其他位置引用它。

```latex
\documentclass{article}

\begin{document}
Hey world!

This is a first document.

\section{Title of the first section}

Text of material for the first section.

\subsection{Subsection of the first section}
\label{subsec:labelone}

Text of material for the first subsection.
\begin{equation}
  e^{i\pi}+1 = 0
\label{eq:labeltwo}
\end{equation}

In subsection~\ref{subsec:labelone} is equation~\ref{eq:labeltwo}.
\end{document}
```

这里有两个 `label{...}` 命令，一个在 subsection 之后，一个在 equation 环境内部。它们在最后一句的 `\ref{...}` 命令中一起出现。当你运行 LaTeX 时，它保存 labels 的信息到辅助文件 `(.aux)` 中。对于 `\label{subsec:labelone}` ，LaTeX 知道现在位于 subsection 环境中，所以保存了 subsection 的编号。对于 `\label{eq:labeltwo}`，LaTeX 知道最新关注的环境是 equation，所以它保存了对应 equation 的信息。当你要求引用时，LaTeX 会在辅助文件中得到它。

`subsec:` 和 `eq:` 并不被 LaTeX 使用；实际上，它只是为了记下来在这个标签之前最近的位置在处理什么内容。明确地写出它们可以帮助你记住标签的含义。

注意引用前的符号带子（`~`）。你或许不想在 `subsection` 和它的编号之间、或者是 `equation` 和它的编号之间产生断行。放一个符号带子表示 LaTeX 不会在这里产生断行。

## 7 数学模式

在数学模式下，空格会被省略，字符间（基本上）都会有恰当的空格。

数学模式有两种形式：

- 行内公式
- 行间公式

```latex
\documentclass{article}
\usepackage[T1]{fontenc}
\begin{document}
A sentence with inline mathematics: $y = mx + c$.
A second sentence with inline mathematics: $5^{2}=3^{2}+4^{2}$.

A second paragraph containing display math.
\[
  y = mx + c
\]
See how the paragraph continues after the display.
\end{document}
```

### 7.1 行内数学模式

1. 由上文可见，行内数学公式是通过一对美元符号（`$...$`）标记的。当然用 `\( ... \)` 标记也是可行的。输入简单表达式不需要任何特殊的标记，你将看到数学公式会被排版出合适的间隔而且文字会变成斜体。
2. 行内数学模式限制了表达式的纵向高度以使数学公式尽可能不打乱段落的行间距。
3. 注意到**所有的**数学符号都应被标记成数学模式下的。即使只是一个字符也应该使用 `... $2$ ...` 而不是 `... 2 ...`。否则，比如说，你需要输入一个负数，然后需要使用数学模式来输入负号，那么 `... $-2$ ...` 可能就会产生与文本模式下不同的数字字体（这取决于文档类的选择）。相反地，要注意从其他地方复制过来的纯文本可能会导致数学模式的构造，比如用于表示金额的美元符号 `$` ，或者是表示文件名的下划线符号 `_`（这些应该分别被标记为 `\$` 和 `\_`）。
4. **我们可以很容易地添加上标和下标：分别通过 `^` 和 `_` 标记。**

```latex
\documentclass{article}
\usepackage[T1]{fontenc}
\begin{document}
Superscripts $a^{b}$ and subscripts $a_{b}$.
\end{document}
```

***LaTeX 里有许多专有的数学模式命令。它们中有些很容易，比如 `\sin` 和 `\log` 用来分别表示正弦符号和对数符号，`\theta` 用于表示对应的希腊符号。***

```latex
\documentclass{article}
\usepackage[T1]{fontenc}
\begin{document}
Some mathematics: $y = 2 \sin \theta^{2}$.
\end{document}
```

### 7.2 行间数学模式

你可以在行间模式中使用与行内模式一样的命令。行间数学模式默认设置在居中位置，并为「仍属于该段」的大型公式所准备。注意到行间数学环境不允许在数学模式中结束该段，所以你也许不能够在该模式的源代码中留有空行

段落都应当在行间公式**之前**开始，所以不要在行间数学环境之前留有空行。如果你的数学公式有多行，不要使用多个连续的行间数学环境（这将导致不一致的行间距），而是使用一种多行行间公式环境，比如稍后提及的 `amsmath` 宏包中 `align` 环境。

```latex
\documentclass{article}
\usepackage[T1]{fontenc}
\begin{document}
A paragraph about a larger equation
\[
\int_{-\infty}^{+\infty} e^{-x^2} \, dx。 %设置积分上下限
\]
\end{document}
```

添加了一个间隙：`\,`，以在 `dx` 前添加一个小间隙。微分符号的格式各有差别：一些出版商使用直立的「d」而其他出版商使用斜体的「*d*」。为了适应这两种方式，编写源代码的一种方法就是定义一个 `\diff` 命令，这样你就能够根据需要进行调整

```latex
\documentclass{article}
\usepackage[T1]{fontenc}
\newcommand{\diff}{\mathop{}\!d}            % 斜体
% \newcommand{\diff}{\mathop{}\!\mathrm{d}} % 直立体
\begin{document}
A paragraph about a larger equation
\[
\int_{-\infty}^{+\infty} e^{-x^2} \diff x
\]
\end{document}
```

通常需要对公式编号，使用 `equation` 环境可以创建这些编号

```latex
\documentclass{article}
\usepackage[T1]{fontenc}
\begin{document}
A paragraph about a larger equation
\begin{equation}
\int_{-\infty}^{+\infty} e^{-x^2} \, dx
\end{equation}
\end{document}
```

### 7.3 [**`amsmath` 宏包**](https://www.latexstudio.net/LearnLaTeX/lesson/10.html#amsmath-%E5%AE%8F%E5%8C%85)

数学符号太多了，就意味着 LaTeX 内核内置的工具并不能提供所有内容。`amsmath` 宏包拓展了内核以提供更多的方案

```latex
\documentclass{article}
\usepackage[T1]{fontenc}
\usepackage{amsmath}

\begin{document}
Solve the following recurrence for $ n,k\geq 0 $:
\begin{align*}
  Q_{n,0} &= 1   \quad Q_{0,k} = [k=0];  \\
  Q_{n,k} &= Q_{n-1,k}+Q_{n-1,k-1}+\binom{n}{k}, \quad\text{for $n$, $k>0$.}
\end{align*}
\end{document}
```

`align*` 环境会让公式在和号 `&` 处对齐，这和表格的用法是一样的。注意一下我们是如何使用 `\quad` 添加小间距、以及使用 `\text` 向数学模式添加正常的文字。我们这里还使用了另一个数学模式命令：二项式 `\binom`。

还注意到我们使用了 `align*` 环境，让公式不被编号。大部分的数学环境都会默认对公式编号，而带星号（`*`）的变式关闭了编号功能。

### 7.4 数学模式的参数

和正常文本不同的是，数学模式的字体变化通常会蕴含非常特定的意义。因此它们经常被写做特殊的命令。下面就是你需要的一组命令：

- `\mathrm`: 罗马字体（直立体）
- `\mathit`: 使用普通文本字间隔的斜体
- `\mathbf`: 粗体
- `\mathsf`: 衬线字体
- `\mathtt`: 等宽字体（打字机字体）
- `\mathbb`: 双重粗体（黑板粗体，由 `amsfonts` 宏包提供）

这些命令接受英文字母作为参数，因此如果我们要写出一个矩阵符号的话：

```latex
\documentclass{article}
\usepackage[T1]{fontenc}
\begin{document}
The matrix $\mathbf{M}$.
\end{document}
```

这些数学字体命令 `\math..` 针对数学使用了特殊的字体。有时你可能需要在数学公式中嵌入一个属于外部句子的词，并需要使用当前文段的字体，在这种情况下可以使用 `\text{...}`（由 `amsmath` 宏包提供）或者是特定的字体比如 `\textrm{..}`。

```latex
\documentclass{article}
\usepackage[T1]{fontenc}
\usepackage{amsmath}
\begin{document}

$\text{bad use } size  \neq \mathit{size} \neq \mathrm{size} $

\textit{$\text{bad use } size \neq \mathit{size} \neq \mathrm{size} $}

\end{document}
```

\textit 表示斜体 但是必须在pdflatex下起作用

## 8 **格式：字体与间距**

在 LaTeX 中输入一个空行将会另起一个段落，表征为以一个首行缩进开始的段落

### 8.1 段间距

一种比较常见的样式是段落首行没有缩进，但要在段落之间插入空行。这种情形我们可以使用 `parskip` 宏包实现。

```latex
\documentclass{article}
\usepackage[T1]{fontenc}
\usepackage[parfill]{parskip}
\usepackage{lipsum} % 为了填充假文
\begin{document}
\lipsum
\end{document}
```

### 8.2 强制断行

部分情况下，你都不应当在 LaTeX 中强制断行：你几乎只是想另起一段，或者是想在段落之间添加空行（就像我们看到的那样，使用 `parskip` 宏包就可实现）。

只有**很少的**情况下你需要使用 `\\` 来另起一行而不另起一段：

- 在表行的结尾处
- 在 `center` 环境中
- 诗歌（`verse` 环境）

如果不属于这些情况，大多数时候你就**不**应当使用 `\\`。

### 8.3 显式添加间距

我们可以使用 `\,` 添加一个小的间距（大概半个空格宽度）。在数学模式中，还可以使用：`\.`，`\:`，`\;`，以及用于添加负间隙的 `\!`。

极少情况下，比如在做一个标题页的时候，你可能需要显式地添加水平间距或垂直间距。对于这种情形，我们可以使用 `\hspace{间距参数}` 和 `\vspace{间距参数}`

### 8.4 显式设置文本格式

大部分情况下使用逻辑结构是更好的。但是有些时候，你可能想要对字体做加粗、斜体、等宽等等的处理。这时就有两种命令可用：一种用于小段，一种用于大段。

对于小段而言，我们可以使用 `\textbf`, `\textit`, `\textrm`, `\textsf`, `\texttt` 以及 `\textsc`。

对于大段来说，我们使用能够更改字体设置的命令：比如说 `\bfseries` 和 `\itshape`。因为这些命令设置完毕后不会复原，所以我们需要定义一个「组（group）」用以防止这些设置作用于整个文档。LaTeX 环境是一个组，表格单元格也是一个组，我们也可以使用 `{...}` 来显式地定义一个组。

我们可以采用类似的方式设置字体大小——这些命令也是持续向后影响的设置。这些命令设置相对字体大小：常见的有 `\huge`, `\large`, `\normalsize`, `\small` 以及 `\footnotesize`。

请注意要在改回字体大小**之前**结束该段：

```latex
\documentclass{article}
\usepackage[T1]{fontenc}
\begin{document}
Normal text.

\begin{center}
{\itshape\large Some text\par}
Normal text
{\bfseries\small Much smaller text\par}
\end{center}

\end{document}
```

note：组的定义以及如何进行设置

# 9 引用与参考文献

对于文献引用来说，虽然你可以直接引用当前文档中的内容，但是大部分情况下，你可能会从一个或多个外部文件得到文献信息。这种文件就是文献数据库，以一种便于处理的格式存储。使用一个或多个文献数据库可以让你复用一些信息并避免手动设定格式。

## 9.1 引用文献库

引用数据库一般被称作「BibTeX 文件」，并带有文件扩展名 `.bib`。它们包含一条或多条记录，每条记录对应一个引用，每条记录中都会有一系列的域（field）。

```latex
@article{Thomas2008,
  author  = {Thomas, Christine M. and Liu, Tianbiao and Hall, Michael B.
             and Darensbourg, Marcetta Y.},
  title   = {Series of Mixed Valent {Fe(II)Fe(I)} Complexes That Model the
             {H(OX)} State of [{FeFe}]Hydrogenase: Redox Properties,
             Density-Functional Theory Investigation, and Reactivity with
             Extrinsic {CO}},
  journal = {Inorg. Chem.},
  year    = {2008},
  volume  = {47},
  number  = {15},
  pages   = {7009-7024},
  doi     = {10.1021/ic800654a},
}
@book{Graham1995,
  author    = {Ronald L. Graham and Donald E. Knuth and Oren Patashnik},
  title     = {Concrete Mathematics},
  publisher = {Addison-Wesley},
  year      = {1995},
}
```

手动编写 `.bib` 可太枯燥了，所以大部分人都会选择一个专用编辑器。[**JabRef**](https://www.jabref.org/) 跨平台并被广泛使用，当然也有很多其他可用软件。

## 9.2 从数据库转化信息

将信息插入你的文档分三步走。

第一步，使用 LaTeX 编译你的文档，这将会产生包含文档引用列表的文件。

第二步，运行一个从引用数据库获取信息的程序，检索出你使用的引用条目，然后按照顺序将它们排列起来。

第三步，再次编译你的文档，这样 LaTeX 就可以使用这些信息来解析你的引用信息。通常这需要至少两次的编译来解析所有的引用。

对于第二步而言，现在有两种广泛使用的系统：BibTeX 和 Biber。Biber 仅供 LaTeX 中一个叫 `biblatex` 的宏包使用，而 BibTeX 可以不搭配宏包使用或者搭配 `natbib` 使用。

引用（citation）和参考（reference）的格式是独立于 BibTeX 数据库的，并且被一种名为 `style`（样式）的东西所设定。我们将会看到这些东西在 BibTeX 和 `biblatex` 工作流中有些许差异，但整体思想是一致的：我们可以选择引用显示的方式。

## 9.2 配合natbib的bibtex工作流

虽然可以在不加载任何宏包的情况下向 LaTeX 文档插入引用，但这还是比较局限的。与之相对的，我们将使用 `natbib` 宏包，允许我们创建不同类型的引用，并且可以使用许多类型的样式。

```latex
\documentclass{article}
\usepackage[T1]{fontenc}
\usepackage{natbib}

\begin{document}
The mathematics showcase is from \citet{Graham1995}, whereas
there is some chemistry in \citet{Thomas2008}.

Some parenthetical citations: \citep{Graham1995}
and then \citep[p.~56]{Thomas2008}.

\citep[See][pp.~45--48]{Graham1995}

Together \citep{Graham1995,Thomas2008}

\bibliographystyle{plainnat}
\bibliography{learnlatex}
\end{document}
```

`natbib` 宏包提供了文本和括号的引用样式：分别是 `\citet` 和 `\citep`。参考文献样式是通过 `\bibliographystyle` 这一行选择的：这里我们使用了 `plainnat` 样式。参考文献实际上是通过 `\bibliography` 这一行插入的，同时也选择了使用的数据库（多个数据库采用逗号分隔数据库名）。

页码引用可以通过一个可选参数添加。如果提供了两个可选参数，那么第一个可选参数就被当作引用前的小标记，第二个可选参数就被当作引用标签后的页码引用。

## 9.3  biblatex 工作流

`biblatex` 宏包和 `natbib` 运作方式稍有不同，主要体现在导言区内选择数据库和文档正文内的排印上。它会有一些新的命令。

```latex
\documentclass{article}
\usepackage[T1]{fontenc}
\usepackage[style=authoryear]{biblatex}
\addbibresource{learnlatex.bib} % 引用文件

\begin{document}
The mathematics showcase is from \autocite{Graham1995}.

Some more complex citations: \parencite{Graham1995} or
\textcite{Thomas2008} or possibly \citetitle{Graham1995}.

\autocite[56]{Thomas2008}

\autocite[See][45-48]{Graham1995}

Together \autocite{Thomas2008,Graham1995}

\printbibliography
\end{document}
```

注意到 `\addbibresource` **需要**数据库文件名的全称，而不是我们对于 `natbib` 下的 `\bibliography` 中省略 `.bib` 的情形。虽然 `biblatex` 使用了相对更长的引用命令，但是这些基本上都是顾名思义的。

再次地，引用前后的文字可以通过可选参数插入。注意，这里的页码不需要添加 `p.~` 或者 `pp.~` 前缀，`biblatex` 可以自动添加合适的前缀。

在 `biblatex` 里，加载宏包时文献样式就被选择了。这里，我们使用了 `authoryear`，当然也可以用 `numeric` 数字编码和其他许多样式。

[LaTeX公式手册(全网最全) - Hexo](https://www.notion.so/LaTeX-Hexo-11f028a90bf181dbb37acda6d123b90d?pvs=21)