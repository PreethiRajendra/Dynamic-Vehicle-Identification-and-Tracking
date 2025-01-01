from django.db import models
from django.contrib.auth import get_user_model
from django.contrib.auth.models import User


User = get_user_model()

class UserProfile(models.Model):
    MALE = 'M'
    FEMALE = 'F'
    TRANS = 'T'
    GENDER_CHOICES = [
        (MALE, 'Male'),
        (FEMALE, 'Female'),
        (TRANS, 'Transgender'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
    age = models.IntegerField(null = True)
    height = models.IntegerField(null = True)
    weight = models.IntegerField(null = True)

    def __str__(self):
        return self.user.username


class UploadedImage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='uploaded_images/')
    plate_text = models.CharField(max_length=255, blank=True, null=True)
    state = models.CharField(max_length=100, blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)


class UploadHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file_name = models.CharField(max_length=255)
    file_type = models.CharField(max_length=50)
    upload_date = models.DateTimeField(auto_now_add=True)
    detected_plate_numbers = models.TextField()  # Add this if not present
    detected_states = models.TextField()        # For state info (optional)

    def __str__(self):
        return self.file_name