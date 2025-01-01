
from django import forms
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError

class ImageUploadForm(forms.Form):
    image = forms.ImageField(label="Select an image")




class VideoUploadForm(forms.Form):
    video = forms.FileField(label="Upload a video file")

class CustomAuthenticationForm(forms.Form):
    username_or_email = forms.CharField(
        label="Username or Email",
        widget=forms.TextInput(attrs={'placeholder': 'Username or Email', 'class': 'form-control'})
    )
    password = forms.CharField(
        label="Password",
        widget=forms.PasswordInput(attrs={'placeholder': 'Password', 'class': 'form-control'})
    )

"""
class ForgotPasswordForm(forms.Form):
    email = forms.EmailField(label='Email Address', required=True)
    
    def clean_email(self):
        email = self.cleaned_data.get('email')
        if not User.objects.filter(email=email).exists():
            raise ValidationError("No account with this email exists.")
        return email
    
"""

class ForgotPasswordForm(forms.Form):
    email = forms.EmailField(label='Enter your email address', max_length=100)
    new_password = forms.CharField(widget=forms.PasswordInput, label='Enter new password')
    confirm_password = forms.CharField(widget=forms.PasswordInput, label='Confirm new password')