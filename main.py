import fire

class Styler:

  def transform(self, input, output, model, metadata):
    print('Transforming data from {} to {} using model {} with metadata {}'.format(input, output, model, metadata))

if __name__ == '__main__':
  fire.Fire(Styler)