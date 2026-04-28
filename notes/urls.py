from django.urls import path
from . import views

app_name = 'notes'

urlpatterns = [
    path('', views.home, name='home'),
    path('search/', views.search, name='search'),
    path('api/calendar-events/', views.calendar_events, name='calendar_events'),
    path('api/upload-image/', views.upload_image, name='upload_image'),

    # Category
    path('category/new/', views.category_create, name='category_create'),
    path('category/<int:pk>/', views.category_detail, name='category_detail'),
    path('category/<int:pk>/edit/', views.category_edit, name='category_edit'),
    path('category/<int:pk>/delete/', views.category_delete, name='category_delete'),

    # SubCategory
    path('category/<int:category_pk>/sub/new/', views.subcategory_create, name='subcategory_create'),
    path('sub/<int:pk>/edit/', views.subcategory_edit, name='subcategory_edit'),
    path('sub/<int:pk>/delete/', views.subcategory_delete, name='subcategory_delete'),

    # Article
    path('sub/<int:subcategory_pk>/article/new/', views.article_create, name='article_create'),
    path('article/<int:pk>/', views.article_detail, name='article_detail'),
    path('article/<int:pk>/edit/', views.article_edit, name='article_edit'),
    path('article/<int:pk>/delete/', views.article_delete, name='article_delete'),
]
