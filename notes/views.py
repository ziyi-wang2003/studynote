import os
import json
from datetime import date, timedelta
from django.conf import settings
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse
from django.db.models import Q, Count
from django.views.decorators.http import require_POST
from django.contrib.auth.decorators import login_required
from .models import Category, SubCategory, Article
from .forms import CategoryForm, SubCategoryForm, ArticleForm


def home(request):
    categories = Category.objects.prefetch_related('subcategories__articles').all()
    recent_articles = Article.objects.select_related('subcategory__category').order_by('-updated_at')[:10]
    total_articles = Article.objects.count()
    total_categories = Category.objects.count()
    context = {
        'categories': categories,
        'recent_articles': recent_articles,
        'total_articles': total_articles,
        'total_categories': total_categories,
    }
    return render(request, 'home.html', context)


def category_detail(request, pk):
    category = get_object_or_404(Category.objects.prefetch_related('subcategories__articles'), pk=pk)
    categories = Category.objects.all()
    context = {
        'category': category,
        'categories': categories,
    }
    return render(request, 'category_detail.html', context)


def article_detail(request, pk):
    article = get_object_or_404(Article.objects.select_related('subcategory__category'), pk=pk)
    category = article.category
    siblings = Article.objects.filter(subcategory=article.subcategory).exclude(pk=pk)
    context = {
        'article': article,
        'category': category,
        'siblings': siblings,
    }
    return render(request, 'article_detail.html', context)


@login_required
def category_create(request):
    if request.method == 'POST':
        form = CategoryForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('notes:home')
    else:
        form = CategoryForm()
    return render(request, 'category_form.html', {'form': form, 'title': '新建大分区'})


@login_required
def category_edit(request, pk):
    category = get_object_or_404(Category, pk=pk)
    if request.method == 'POST':
        form = CategoryForm(request.POST, instance=category)
        if form.is_valid():
            form.save()
            return redirect('notes:category_detail', pk=pk)
    else:
        form = CategoryForm(instance=category)
    return render(request, 'category_form.html', {'form': form, 'title': f'编辑分区: {category.name}'})


@login_required
def category_delete(request, pk):
    category = get_object_or_404(Category, pk=pk)
    if request.method == 'POST':
        category.delete()
        return redirect('notes:home')
    return render(request, 'confirm_delete.html', {'object': category, 'type': '大分区'})


@login_required
def subcategory_create(request, category_pk):
    category = get_object_or_404(Category, pk=category_pk)
    if request.method == 'POST':
        form = SubCategoryForm(request.POST)
        if form.is_valid():
            sub = form.save(commit=False)
            sub.category = category
            sub.save()
            return redirect('notes:category_detail', pk=category_pk)
    else:
        form = SubCategoryForm()
    return render(request, 'subcategory_form.html', {
        'form': form, 'category': category, 'title': f'新建小分区 - {category.name}'
    })


@login_required
def subcategory_edit(request, pk):
    sub = get_object_or_404(SubCategory.objects.select_related('category'), pk=pk)
    if request.method == 'POST':
        form = SubCategoryForm(request.POST, instance=sub)
        if form.is_valid():
            form.save()
            return redirect('notes:category_detail', pk=sub.category.pk)
    else:
        form = SubCategoryForm(instance=sub)
    return render(request, 'subcategory_form.html', {
        'form': form, 'category': sub.category, 'title': f'编辑小分区: {sub.name}'
    })


@login_required
def subcategory_delete(request, pk):
    sub = get_object_or_404(SubCategory.objects.select_related('category'), pk=pk)
    category_pk = sub.category.pk
    if request.method == 'POST':
        sub.delete()
        return redirect('notes:category_detail', pk=category_pk)
    return render(request, 'confirm_delete.html', {'object': sub, 'type': '小分区'})


@login_required
def article_create(request, subcategory_pk):
    sub = get_object_or_404(SubCategory.objects.select_related('category'), pk=subcategory_pk)
    if request.method == 'POST':
        form = ArticleForm(request.POST)
        if form.is_valid():
            article = form.save(commit=False)
            article.subcategory = sub
            article.save()
            return redirect('notes:article_detail', pk=article.pk)
    else:
        form = ArticleForm(initial={'subcategory': sub})
    form.fields['subcategory'].queryset = SubCategory.objects.filter(category=sub.category)
    return render(request, 'article_form.html', {
        'form': form, 'category': sub.category, 'title': '新建文章', 'is_new': True
    })


@login_required
def article_edit(request, pk):
    article = get_object_or_404(Article.objects.select_related('subcategory__category'), pk=pk)
    if request.method == 'POST':
        form = ArticleForm(request.POST, instance=article)
        if form.is_valid():
            form.save()
            return redirect('notes:article_detail', pk=pk)
    else:
        form = ArticleForm(instance=article)
    form.fields['subcategory'].queryset = SubCategory.objects.filter(category=article.category)
    return render(request, 'article_form.html', {
        'form': form, 'category': article.category, 'title': f'编辑: {article.title}', 'is_new': False
    })


@login_required
def article_delete(request, pk):
    article = get_object_or_404(Article.objects.select_related('subcategory__category'), pk=pk)
    sub_pk = article.subcategory.pk
    category_pk = article.category.pk
    if request.method == 'POST':
        article.delete()
        return redirect('notes:category_detail', pk=category_pk)
    return render(request, 'confirm_delete.html', {'object': article, 'type': '文章'})


def search(request):
    query = request.GET.get('q', '').strip()
    results = []
    if query:
        results = Article.objects.select_related('subcategory__category').filter(
            Q(title__icontains=query) | Q(content__icontains=query)
        )[:50]
    return render(request, 'search.html', {'query': query, 'results': results})


def calendar_events(request):
    """API endpoint: return article events for the calendar."""
    articles = Article.objects.select_related('subcategory__category').all()
    events = []
    for a in articles:
        events.append({
            'title': a.title,
            'start': a.created_at.strftime('%Y-%m-%d'),
            'url': a.get_absolute_url(),
            'color': a.category.color,
        })
        if a.updated_at.date() != a.created_at.date():
            events.append({
                'title': f'[更新] {a.title}',
                'start': a.updated_at.strftime('%Y-%m-%d'),
                'url': a.get_absolute_url(),
                'color': '#43a047',
            })
    return JsonResponse(events, safe=False)


@login_required
@require_POST
def upload_image(request):
    """Handle image upload from the editor. Returns markdown image syntax."""
    f = request.FILES.get('image')
    if not f:
        return JsonResponse({'error': '未选择文件'}, status=400)

    allowed = ('.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp')
    ext = os.path.splitext(f.name)[1].lower()
    if ext not in allowed:
        return JsonResponse({'error': '不支持的图片格式'}, status=400)

    upload_dir = os.path.join(settings.MEDIA_ROOT, 'images', 'uploads')
    os.makedirs(upload_dir, exist_ok=True)

    filepath = os.path.join(upload_dir, f.name)
    # Avoid overwriting
    base, extension = os.path.splitext(f.name)
    counter = 1
    while os.path.exists(filepath):
        filepath = os.path.join(upload_dir, f'{base}_{counter}{extension}')
        counter += 1

    with open(filepath, 'wb+') as dest:
        for chunk in f.chunks():
            dest.write(chunk)

    relative = os.path.relpath(filepath, settings.MEDIA_ROOT)
    url = settings.MEDIA_URL + relative
    return JsonResponse({'url': url, 'markdown': f'![{f.name}]({url})'})
