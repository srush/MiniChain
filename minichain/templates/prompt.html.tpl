
<!-- <link rel="stylesheet" href="https://cdn.rawgit.com/Chalarangelo/mini.css/v3.0.1/dist/mini-default.min.css"> -->
 <main class="container">

<h3>{{name}}</h3>

<dl>
  <dt>Input:</dt>
  <dd>
{% highlight 'python' %}
{{input}}
{% endhighlight %}

  </dd>

  <dt> Full Prompt: </dt>
  <dd>
    <details>
      <summary>Prompt</summary>
      <p>{{prompt | safe}}</p>
    </details>
  </dd>

  <dt> Response: </dt>
  <dd>
    {{response | replace("\n", "<br>")  | safe}}
  </dd>

  <dt>Value:</dt>
  <dd>
{% highlight 'python' %}
{{output}}
{% endhighlight %}
  </dd>
</main>

