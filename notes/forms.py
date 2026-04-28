from django import forms
from .models import Category, SubCategory, Article


class CategoryForm(forms.ModelForm):
    class Meta:
        model = Category
        fields = ['name', 'description', 'icon', 'color', 'order']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': '例如：算法学习'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 3, 'placeholder': '简要描述此分区的内容'}),
            'icon': forms.Select(attrs={'class': 'form-select'}, choices=[
                ('bi-folder', 'Folder 文件夹'),
                ('bi-book', 'Book 书本'),
                ('bi-code-slash', 'Code 代码'),
                ('bi-cpu', 'CPU 芯片'),
                ('bi-graph-up', 'Graph 图表'),
                ('bi-lightbulb', 'Lightbulb 灯泡'),
                ('bi-pencil-square', 'Pencil 铅笔'),
                ('bi-journal-text', 'Journal 日记'),
                ('bi-robot', 'Robot 机器人'),
                ('bi-braces', 'Braces 大括号'),
                ('bi-diagram-3', 'Diagram 图形'),
                ('bi-mortarboard', 'Mortarboard 学士帽'),
            ]),
            'color': forms.TextInput(attrs={'class': 'form-control form-control-color', 'type': 'color'}),
            'order': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': '数字越小越靠前'}),
        }


class SubCategoryForm(forms.ModelForm):
    class Meta:
        model = SubCategory
        fields = ['name', 'order']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': '小分区名称'}),
            'order': forms.NumberInput(attrs={'class': 'form-control'}),
        }


class ArticleForm(forms.ModelForm):
    class Meta:
        model = Article
        fields = ['title', 'summary', 'subcategory', 'content', 'order', 'is_pinned']
        widgets = {
            'title': forms.TextInput(attrs={'class': 'form-control', 'placeholder': '文章标题'}),
            'summary': forms.TextInput(attrs={'class': 'form-control', 'placeholder': '一句话描述这篇文章（显示在列表卡片上）'}),
            'subcategory': forms.Select(attrs={'class': 'form-select'}),
            'content': forms.Textarea(attrs={'class': 'form-control markdown-editor', 'rows': 20, 'placeholder': '在此输入 Markdown 内容...'}),
            'order': forms.NumberInput(attrs={'class': 'form-control'}),
            'is_pinned': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }
