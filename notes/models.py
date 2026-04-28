from django.db import models
from django.urls import reverse


class Category(models.Model):
    name = models.CharField('分区名称', max_length=100)
    description = models.TextField('简介', blank=True)
    icon = models.CharField('图标类名', max_length=50, default='bi-folder')
    color = models.CharField('主题色', max_length=20, default='#1565c0')
    order = models.IntegerField('排序', default=0)
    created_at = models.DateTimeField('创建时间', auto_now_add=True)
    updated_at = models.DateTimeField('更新时间', auto_now=True)

    class Meta:
        verbose_name = '大分区'
        verbose_name_plural = '大分区'
        ordering = ['order', '-created_at']

    def __str__(self):
        return self.name

    def get_absolute_url(self):
        return reverse('notes:category_detail', args=[self.pk])

    def article_count(self):
        return Article.objects.filter(subcategory__category=self).count()


class SubCategory(models.Model):
    category = models.ForeignKey(
        Category, on_delete=models.CASCADE, related_name='subcategories',
        verbose_name='所属分区'
    )
    name = models.CharField('小分区名称', max_length=100)
    order = models.IntegerField('排序', default=0)
    created_at = models.DateTimeField('创建时间', auto_now_add=True)
    updated_at = models.DateTimeField('更新时间', auto_now=True)

    class Meta:
        verbose_name = '小分区'
        verbose_name_plural = '小分区'
        ordering = ['order', '-created_at']

    def __str__(self):
        return f'{self.category.name} / {self.name}'

    def get_absolute_url(self):
        return reverse('notes:category_detail', args=[self.category.pk]) + f'#sub-{self.pk}'


class Article(models.Model):
    subcategory = models.ForeignKey(
        SubCategory, on_delete=models.CASCADE, related_name='articles',
        verbose_name='所属小分区'
    )
    title = models.CharField('标题', max_length=200)
    summary = models.CharField('简介', max_length=300, blank=True, help_text='卡片上显示的简短描述，留空则不显示')
    content = models.TextField('内容 (Markdown)')
    order = models.IntegerField('排序', default=0)
    is_pinned = models.BooleanField('置顶', default=False)
    created_at = models.DateTimeField('创建时间', auto_now_add=True)
    updated_at = models.DateTimeField('更新时间', auto_now=True)

    class Meta:
        verbose_name = '文章'
        verbose_name_plural = '文章'
        ordering = ['-is_pinned', 'order', '-created_at']

    def __str__(self):
        return self.title

    def get_absolute_url(self):
        return reverse('notes:article_detail', args=[self.pk])

    @property
    def category(self):
        return self.subcategory.category
